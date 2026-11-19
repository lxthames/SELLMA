"""Separate follow-up to the inspected original experiment; run from any directory."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
ORIGINAL_FILES = ['experiment.py', 'protocol.json', 'data/benchmark.json',
                  'results/summary.json', 'results/examples.csv']


def hashes(names):
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_inputs():
    data = json.loads((ROOT / 'data/followup.json').read_text(encoding='utf-8'))
    protocol = json.loads((ROOT / 'followup_protocol.json').read_text(encoding='utf-8'))
    original = json.loads((ROOT / 'data/benchmark.json').read_text(encoding='utf-8'))
    old_nouns = {n[0].lower() for n in original['nouns']}
    nouns = data['nouns']
    assert len(nouns) == 48 and len({n['lemma'] for n in nouns}) == 48
    assert not old_nouns.intersection(n['definite'] for n in nouns)
    for split, count in [('dev', 8), ('test', 16)]:
        for gender in ('common', 'neuter'):
            assert sum(n['split'] == split and n['gender'] == gender for n in nouns) == count
    for field in ('lemma', 'definite', 'family'):
        dev = {n[field] for n in nouns if n['split'] == 'dev'}
        test = {n[field] for n in nouns if n['split'] == 'test'}
        assert dev.isdisjoint(test), field
    rows = []
    for noun in nouns:
        common = noun['gender'] == 'common'
        context = dict(noun, article='en' if common else 'ett',
                       definite_title=noun['definite'].capitalize())
        for task, frames in data['frames'].items():
            pair = ('den', 'det') if task == 'source' else ('min', 'mitt')
            for frame_id, frame in enumerate(frames):
                rows.append(dict(id=f"{noun['lemma']}_{task}_{frame_id}", noun=noun['lemma'],
                                 gender=noun['gender'], split=noun['split'], task=task,
                                 frame=frame_id, prompt=frame.format(**context),
                                 expected=pair[0] if common else pair[1]))
    for index, (prompt, expected) in enumerate(original['regressions']):
        rows.append(dict(id=f'regression_{index}', noun='', gender='', split='regression',
                         task='regression', frame=0, prompt=prompt, expected=expected))
    assert len({r['id'] for r in rows}) == len(rows)
    assert all('?' not in r['prompt'] for r in rows)
    return data, protocol, rows


def correctness(margin, signs):
    signed = margin * signs
    return (signed > 0).float() + (signed == 0).float() / 2


def select_settings(margin, shift_at_unit_strength, signs, rows, protocol):
    # Slice before scoring: held-out margins cannot affect either selection.
    dev = torch.tensor([r['split'] == 'dev' and r['task'] == 'target' for r in rows])
    margin, shift, signs = margin[dev], shift_at_unit_strength[dev], signs[dev]
    candidates = []
    for strength in protocol['strengths']:
        score = correctness(margin + shift * strength, signs).mean().item()
        candidates.append(dict(strength=strength, accuracy=score))
    selected = min(candidates, key=lambda item: (-item['accuracy'], item['strength']))
    grid = protocol['calibration_grid']
    biases = np.arange(grid['min'], grid['max'] + grid['step'] / 2, grid['step'])
    scores = [correctness(margin + float(bias), signs).mean().item() for bias in biases]
    best = max(scores)
    bias = min((float(b) for b, score in zip(biases, scores) if score == best),
               key=lambda b: (abs(b), b))
    return dict(selected_strength=selected['strength'], development_candidates=candidates,
                calibration_bias=bias, calibration_development_accuracy=best,
                calibration_at_grid_boundary=abs(bias) >= 19.99)


def paired_interval(delta, rows, mask, draws, seed):
    rng = np.random.default_rng(seed)
    resampled = []
    for gender in ('common', 'neuter'):
        nouns = list(dict.fromkeys(r['noun'] for i, r in enumerate(rows)
                                  if mask[i] and r['gender'] == gender))
        values = np.array([delta[[i for i, r in enumerate(rows)
                                 if mask[i] and r['noun'] == noun]].mean().item() for noun in nouns])
        samples = rng.integers(0, len(values), (draws, len(values)))
        resampled.append(values[samples].mean(axis=1))
    return np.quantile(np.mean(resampled, axis=0), [0.025, 0.975]).tolist()


@torch.inference_mode()
def collect(protocol, rows, device):
    tokenizer = AutoTokenizer.from_pretrained(protocol['model'], revision=protocol['revision'])
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    ids = {}
    for word in ('den', 'det', 'min', 'mitt'):
        tokens = tokenizer.encode(' ' + word, add_special_tokens=False)
        assert len(tokens) == 1, (word, tokens)
        ids[word] = tokens[0]
    for row in rows:
        pair = ('den', 'det') if row['task'] == 'source' else ('min', 'mitt')
        prefix = tokenizer.encode(row['prompt'], add_special_tokens=False)
        for word in pair:
            assert tokenizer.encode(row['prompt'] + ' ' + word, add_special_tokens=False) == prefix + [ids[word]]
    model = AutoModelForCausalLM.from_pretrained(
        protocol['model'], revision=protocol['revision'], dtype=torch.float32,
        attn_implementation='eager').to(device).eval()
    assert model.get_input_embeddings().weight.data_ptr() == model.lm_head.weight.data_ptr()
    states, logits = [], []
    for start in range(0, len(rows), 8):
        batch = tokenizer([r['prompt'] for r in rows[start:start + 8]],
                          padding=True, return_tensors='pt').to(device)
        hidden = model.model(**batch, use_cache=False).last_hidden_state[:, -1, :]
        states.append(hidden.cpu())
        logits.append(model.lm_head(hidden).cpu())
    hidden, baseline = torch.cat(states), torch.cat(logits)
    weights = model.lm_head.weight.detach().cpu()
    direction = weights[ids['den']] - weights[ids['det']]
    direction = direction / direction.norm() * (weights[ids['min']] - weights[ids['mitt']]).norm()

    # Independent end-to-end checks span task, split, gender, and frame strata.
    representatives = {}
    for i, row in enumerate(rows):
        key = (row['task'], row['split'], row['gender'], row['frame'],
               row['expected'] if row['task'] == 'regression' else '')
        representatives.setdefault(key, i)
    indices = list(representatives.values())
    original_head = model.lm_head
    model.set_output_embeddings(copy.deepcopy(original_head))
    errors = {}
    try:
        for strength in (-max(protocol['strengths']), protocol['original_edit_strength'],
                         max(protocol['strengths'])):
            model.lm_head.weight.copy_(original_head.weight)
            model.lm_head.weight[ids['min']].add_(direction.to(device) * strength / 2)
            model.lm_head.weight[ids['mitt']].sub_(direction.to(device) * strength / 2)
            assert torch.equal(model.get_input_embeddings().weight, original_head.weight)
            error = 0.0
            for start in range(0, len(indices), 8):
                selected = indices[start:start + 8]
                batch = tokenizer([rows[i]['prompt'] for i in selected], padding=True,
                                  return_tensors='pt').to(device)
                direct = model(**batch, use_cache=False).logits[:, -1, :].cpu()
                expected = baseline[selected].clone()
                shift = hidden[selected] @ direction * strength / 2
                expected[:, ids['min']] += shift
                expected[:, ids['mitt']] -= shift
                error = max(error, (direct - expected).abs().max().item())
            assert error < protocol['direct_tolerance'], (strength, error)
            errors[str(strength)] = error
    finally:
        model.set_output_embeddings(original_head)
    restore_error = 0.0
    for start in range(0, len(indices), 8):
        selected = indices[start:start + 8]
        batch = tokenizer([rows[i]['prompt'] for i in selected], padding=True,
                          return_tensors='pt').to(device)
        restored = model(**batch, use_cache=False).logits[:, -1, :].cpu()
        restore_error = max(restore_error, (restored - baseline[selected]).abs().max().item())
    assert restore_error < protocol['direct_tolerance']
    return dict(hidden=hidden, logits=baseline, direction=direction, ids=ids,
                checks=dict(direct_max_errors=errors, restored_max_error=restore_error,
                            direct_prompt_count=len(indices), input_embeddings_unchanged=True,
                            both_candidate_boundaries_verified=True),
                environment=dict(python=platform.python_version(), torch=torch.__version__,
                                 transformers=transformers.__version__, numpy=np.__version__,
                                 device=device, gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                                 dtype='float32', attention='eager'))


def evaluate(cache, protocol, rows, settings):
    hidden, baseline, ids = cache['hidden'], cache['logits'], cache['ids']
    target_margin = baseline[:, ids['min']] - baseline[:, ids['mitt']]
    source_margin = baseline[:, ids['den']] - baseline[:, ids['det']]
    source = torch.tensor([r['task'] == 'source' for r in rows])
    target_test = torch.tensor([r['task'] == 'target' and r['split'] == 'test' for r in rows])
    signs = torch.tensor([1 if r['expected'] in ('den', 'min') else -1 for r in rows])
    base_margin = torch.where(source, source_margin, target_margin)
    base_correct = correctness(base_margin, signs)
    competence = {}
    for task in ('source', 'target'):
        mask = torch.tensor([r['task'] == task and r['split'] == 'test' for r in rows])
        competence[task] = dict(accuracy=base_correct[mask].mean().item(), count=int(mask.sum()))
        for gender in ('common', 'neuter'):
            group = mask & torch.tensor([r['gender'] == gender for r in rows])
            competence[task][gender + '_accuracy'] = base_correct[group].mean().item()
    gate = all(competence[task][gender + '_accuracy'] >= protocol['competence_threshold']
               for task in ('source', 'target') for gender in ('common', 'neuter'))
    selected = settings['selected_strength']
    unit_shift = hidden @ cache['direction'] / 2
    shifts = dict(baseline=torch.zeros(len(rows)), fixed_edit=unit_shift * protocol['original_edit_strength'],
                  reversed_fixed=-unit_shift * protocol['original_edit_strength'],
                  selected_edit=unit_shift * selected, reversed_selected=-unit_shift * selected,
                  calibration=torch.full((len(rows),), settings['calibration_bias'] / 2))
    for seed in protocol['random_seeds']:
        direction = torch.randn(cache['direction'].shape, generator=torch.Generator().manual_seed(seed))
        direction *= cache['direction'].norm() * selected / direction.norm()
        shifts[f'random_{seed}'] = hidden @ direction / 2
    expected_ids = torch.tensor([ids[r['expected']] for r in rows])
    base_logp = baseline.log_softmax(-1)[torch.arange(len(rows)), expected_ids]
    results, examples = {}, []
    for condition, shift in shifts.items():
        edited = baseline.clone()
        edited[:, ids['min']] += shift
        edited[:, ids['mitt']] -= shift
        margin = torch.where(source, source_margin,
                             edited[:, ids['min']] - edited[:, ids['mitt']])
        correct = correctness(margin, signs)
        logp = edited.log_softmax(-1)[torch.arange(len(rows)), expected_ids]
        delta = correct - base_correct
        result = dict(accuracy=correct[target_test].mean().item(), gain=delta[target_test].mean().item(),
                      gain_ci95=paired_interval(delta, rows, target_test, protocol['bootstrap_samples'], protocol['seed']),
                      fixed=int(((base_correct == 0) & (correct == 1) & target_test).sum()),
                      broken=int(((base_correct == 1) & (correct == 0) & target_test).sum()))
        for gender in ('common', 'neuter'):
            mask = target_test & torch.tensor([r['gender'] == gender for r in rows])
            result[gender + '_accuracy'] = correct[mask].mean().item()
        regression = torch.tensor([r['task'] == 'regression' for r in rows])
        result['regression_mean_logp_change'] = (logp - base_logp)[regression].mean().item()
        result['regression_groups'] = {word: (logp - base_logp)[regression & (expected_ids == ids[word])].mean().item()
                                       for word in ('min', 'mitt')}
        result['frame_accuracy'] = {str(frame): correct[target_test & torch.tensor([r['frame'] == frame for r in rows])].mean().item()
                                    for frame in (0, 1)}
        results[condition] = result
        for i, row in enumerate(rows):
            examples.append(dict(condition=condition, **row, correct=float(correct[i]),
                                 correct_margin=float(margin[i] * signs[i]),
                                 logp_change=float(logp[i] - base_logp[i])))
    chosen = results['selected_edit']
    random95 = float(np.quantile([r['gain'] for k, r in results.items() if k.startswith('random_')], 0.95))
    support = dict(competence_gate=gate, positive_gain=chosen['gain'] > 0,
                   positive_gain_interval=chosen['gain_ci95'][0] > 0,
                   beats_calibration=chosen['gain'] > results['calibration']['gain'],
                   beats_random95=chosen['gain'] > random95,
                   regression_threshold=chosen['regression_mean_logp_change'] >= protocol['regression_min_mean_logp_change'])
    return dict(protocol=protocol, selection=settings, competence=competence,
                support_checks=support, supports_transfer=all(support.values()),
                random_gain_percentile95=random95, conditions=results,
                checks=cache['checks'], environment=cache['environment']), examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--reuse-cache', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/followup')
    args = parser.parse_args()
    _, protocol, rows = load_inputs()
    fingerprint = hashes(['followup.py', 'followup_protocol.json', 'data/followup.json'])
    original = hashes(ORIGINAL_FILES)
    if args.validate_only:
        print(json.dumps(dict(nouns=48, source_examples=96, target_examples=96,
                              regression_examples=20, fingerprint=fingerprint), indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'design_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['fingerprints'] != fingerprint or manifest['original_files'] != original:
            raise SystemExit('Recorded design or original artifacts changed; use a new --output directory.')
    else:
        manifest = dict(recorded_before_evaluation_utc=datetime.now(timezone.utc).isoformat(),
                        registration='Local record only; not an independent preregistration.',
                        fingerprints=fingerprint, original_files=original)
        save_json(manifest_path, manifest)
    torch.manual_seed(protocol['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    started = time.perf_counter()
    cache_path = args.output / 'cache.pt'
    if args.reuse_cache:
        cache = torch.load(cache_path, weights_only=False)
        assert cache['fingerprints'] == fingerprint
    else:
        cache = collect(protocol, rows, args.device)
        cache['fingerprints'] = fingerprint
        torch.save(cache, cache_path)
    ids = cache['ids']
    margin = cache['logits'][:, ids['min']] - cache['logits'][:, ids['mitt']]
    signs = torch.tensor([1 if r['expected'] == 'min' else -1 for r in rows])
    settings = select_settings(margin, cache['hidden'] @ cache['direction'], signs, rows, protocol)
    selection_path = args.output / 'selection.json'
    if selection_path.exists():
        assert json.loads(selection_path.read_text(encoding='utf-8')) == settings
    else:
        save_json(selection_path, settings)  # Written before held-out scoring.
    summary, examples = evaluate(cache, protocol, rows, settings)
    summary.update(elapsed_seconds=time.perf_counter() - started, design_manifest=manifest)
    assert hashes(ORIGINAL_FILES) == original, 'Original artifacts changed during follow-up.'
    save_json(args.output / 'summary.json', summary)
    with (args.output / 'examples.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(examples[0]))
        writer.writeheader()
        writer.writerows(examples)
    print(json.dumps(dict(output=str(args.output), selection=settings,
                          competence=summary['competence'], support_checks=summary['support_checks'],
                          elapsed_seconds=summary['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
