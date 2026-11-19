"""Frozen Swedish agreement experiment. Run from the repository root."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent


def load_inputs():
    data = json.loads((ROOT / 'data/benchmark.json').read_text(encoding='utf-8'))
    protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    nouns = data['nouns']
    assert len(nouns) == 60 and len({n[0].lower() for n in nouns}) == 60
    for split, count in [('dev', 6), ('test', 24)]:
        for gender in ['common', 'neuter']:
            assert sum(n[1:] == [gender, split] for n in nouns) == count
    rows = []
    for noun, gender, split in nouns:
        for frame_id, frame in enumerate(data['frames']):
            rows.append(dict(id=f'{noun}_{frame_id}', noun=noun, gender=gender,
                             split=split, prompt=frame.format(noun=noun, noun_lower=noun.lower()),
                             expected='min' if gender == 'common' else 'mitt'))
    for index, (prompt, expected) in enumerate(data['regressions']):
        rows.append(dict(id=f'regression_{index}', noun='', gender='', split='regression',
                         prompt=prompt, expected=expected))
    assert len(data['regressions']) == 20
    return data, protocol, rows


def fingerprint():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ['data/benchmark.json', 'protocol.json', 'experiment.py']}


@torch.inference_mode()
def collect(protocol, rows, device):
    tokenizer = AutoTokenizer.from_pretrained(protocol['model'], revision=protocol['revision'])
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    ids = {}
    for word in ['den', 'det', 'min', 'mitt']:
        tokens = tokenizer.encode(' ' + word, add_special_tokens=False)
        assert len(tokens) == 1, (word, tokens)
        ids[word] = tokens[0]
    model = AutoModelForCausalLM.from_pretrained(
        protocol['model'], revision=protocol['revision'], dtype=torch.float32,
        attn_implementation='eager').to(device).eval()
    assert model.get_input_embeddings().weight.data_ptr() == model.lm_head.weight.data_ptr()
    states, logits = [], []
    for start in range(0, len(rows), 8):
        batch_rows = rows[start:start + 8]
        batch = tokenizer([r['prompt'] for r in batch_rows], padding=True, return_tensors='pt').to(device)
        for index, row in enumerate(batch_rows):
            prefix = tokenizer.encode(row['prompt'], add_special_tokens=False)
            extended = tokenizer.encode(row['prompt'] + ' ' + row['expected'], add_special_tokens=False)
            assert extended == prefix + [ids[row['expected']]], row
        hidden = model.model(**batch, use_cache=False).last_hidden_state[:, -1, :]
        states.append(hidden.cpu())
        logits.append(model.lm_head(hidden).cpu())
    hidden = torch.cat(states)
    baseline = torch.cat(logits)
    weights = model.lm_head.weight.detach().cpu()
    direction = weights[ids['den']] - weights[ids['det']]
    norm = (weights[ids['min']] - weights[ids['mitt']]).norm() * protocol['edit_strength']
    direction = direction / direction.norm() * norm

    # Validate the cached calculation against a real, independent output projection.
    batch = tokenizer([r['prompt'] for r in rows[:2]], padding=True, return_tensors='pt').to(device)
    input_before = model.get_input_embeddings().weight.detach().clone()
    original_head = model.lm_head
    model.set_output_embeddings(copy.deepcopy(original_head))
    model.lm_head.weight[ids['min']].add_(direction.to(device) / 2)
    model.lm_head.weight[ids['mitt']].sub_(direction.to(device) / 2)
    assert torch.equal(input_before, model.get_input_embeddings().weight)
    direct = model(**batch, use_cache=False).logits[:, -1, :].cpu()
    cached = baseline[:2].clone()
    shift = hidden[:2] @ direction / 2
    cached[:, ids['min']] += shift
    cached[:, ids['mitt']] -= shift
    error = (direct - cached).abs().max().item()
    assert error < 1e-4, error
    model.set_output_embeddings(original_head)
    restored = model(**batch, use_cache=False).logits[:, -1, :].cpu()
    restore_error = (restored - baseline[:2]).abs().max().item()
    assert restore_error < 1e-4, restore_error
    return dict(hidden=hidden, logits=baseline, direction=direction, ids=ids,
                fingerprint=fingerprint(), checks=dict(cached_direct_max_error=error,
                restored_baseline_max_error=restore_error, input_embeddings_unchanged=True,
                token_boundaries_verified=True),
                environment=dict(python=platform.python_version(), torch=torch.__version__,
                transformers=transformers.__version__, numpy=np.__version__, device=device,
                gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                dtype='float32', attention='eager'))


def evaluate(cache, protocol, rows):
    hidden, baseline, ids = cache['hidden'], cache['logits'], cache['ids']
    baseline_margin = baseline[:, ids['min']] - baseline[:, ids['mitt']]
    dev = torch.tensor([r['split'] == 'dev' for r in rows])
    test = torch.tensor([r['split'] == 'test' for r in rows])
    regression = torch.tensor([r['split'] == 'regression' for r in rows])
    signs = torch.tensor([1 if r['expected'] == 'min' else -1 for r in rows])

    def accuracy(margins):
        signed = margins * signs
        return (signed > 0).float() + (signed == 0).float() / 2

    grid = protocol['calibration_grid']
    biases = np.arange(grid['min'], grid['max'] + grid['step'] / 2, grid['step'])
    scores = np.array([accuracy(baseline_margin + float(b))[dev].mean().item() for b in biases])
    best = biases[scores == scores.max()]
    bias = float(best[np.argmin(np.abs(best))])
    shifts = {'baseline': torch.zeros(len(rows)),
              'fixed_edit': hidden @ cache['direction'] / 2,
              'calibration': torch.full((len(rows),), bias / 2)}
    for seed in protocol['random_seeds']:
        generator = torch.Generator().manual_seed(seed)
        direction = torch.randn(cache['direction'].shape, generator=generator)
        direction *= cache['direction'].norm() / direction.norm()
        shifts[f'random_{seed}'] = hidden @ direction / 2
    base_accuracy = accuracy(baseline_margin)
    base_logp = baseline.log_softmax(-1)
    expected_ids = torch.tensor([ids[r['expected']] for r in rows])
    expected_baseline = base_logp[torch.arange(len(rows)), expected_ids]
    noun_indices = [[i for i, r in enumerate(rows) if r['noun'] == noun and test[i]]
                    for noun in dict.fromkeys(r['noun'] for r in rows if r['split'] == 'test')]
    rng = np.random.default_rng(protocol['seed'])
    samples = rng.integers(0, len(noun_indices), (protocol['bootstrap_samples'], len(noun_indices)))
    results, examples = {}, []
    for name, shift in shifts.items():
        edited = baseline.clone()
        edited[:, ids['min']] += shift
        edited[:, ids['mitt']] -= shift
        margin = edited[:, ids['min']] - edited[:, ids['mitt']]
        correct = accuracy(margin)
        expected_logp = edited.log_softmax(-1)[torch.arange(len(rows)), expected_ids]
        delta = correct - base_accuracy
        noun_delta = np.array([delta[index].mean().item() for index in noun_indices])
        interval = np.quantile(noun_delta[samples].mean(1), [0.025, 0.975]).tolist()
        result = dict(accuracy=correct[test].mean().item(),
                      gain=delta[test].mean().item(), gain_ci95=interval,
                      fixed=int(((base_accuracy == 0) & (correct == 1) & test).sum()),
                      broken=int(((base_accuracy == 1) & (correct == 0) & test).sum()),
                      mean_correct_margin=(margin * signs)[test].mean().item(),
                      regression_mean_logp_change=(expected_logp - expected_baseline)[regression].mean().item())
        for gender in ['common', 'neuter']:
            mask = test & torch.tensor([r['gender'] == gender for r in rows])
            result[f'{gender}_accuracy'] = correct[mask].mean().item()
        results[name] = result
        for index, row in enumerate(rows):
            examples.append(dict(condition=name, **row, correct=float(correct[index]),
                                 baseline_correct=float(base_accuracy[index]),
                                 correct_margin=float(margin[index] * signs[index]),
                                 expected_logp=float(expected_logp[index]),
                                 logp_change=float(expected_logp[index] - expected_baseline[index])))
    random_gains = [r['gain'] for name, r in results.items() if name.startswith('random_')]
    random95 = float(np.quantile(random_gains, 0.95))
    fixed = results['fixed_edit']
    recommend = (fixed['gain_ci95'][0] > 0 and fixed['gain'] > results['calibration']['gain']
                 and fixed['gain'] > random95 and fixed['regression_mean_logp_change'] >= -0.05)
    return dict(protocol=protocol, fingerprints=cache['fingerprint'], checks=cache['checks'],
                environment=cache['environment'], calibration_bias=bias,
                calibration_at_grid_boundary=abs(bias) >= 19.99,
                random_gain_percentile95=random95, recommend_replication=bool(recommend),
                conditions=results), examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--reuse-cache', action='store_true')
    args = parser.parse_args()
    _, protocol, rows = load_inputs()
    if args.validate_only:
        print(json.dumps(dict(nouns=60, target_examples=120, regression_examples=20, fingerprints=fingerprint()), indent=2))
        return
    torch.manual_seed(protocol['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    started = time.perf_counter()
    cache_path = results / 'cache.pt'
    if args.reuse_cache:
        cache = torch.load(cache_path, weights_only=False)
        assert cache['fingerprint'] == fingerprint(), 'Inputs or implementation changed. Recompute cache.'
    else:
        cache = collect(protocol, rows, args.device)
        torch.save(cache, cache_path)
    summary, examples = evaluate(cache, protocol, rows)
    summary['elapsed_seconds'] = time.perf_counter() - started
    (results / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (results / 'examples.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(examples[0]))
        writer.writeheader()
        writer.writerows(examples)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
