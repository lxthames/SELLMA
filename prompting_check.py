"""Exploratory few-shot comparison on the previously inspected target split."""
import argparse
import csv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from checkpoint_check import CHECKPOINTS
from followup import ROOT, correctness, hashes, load_inputs, paired_interval, save_json


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-chat", action="store_true", help="Also score the same completions with the chat template.")
    args = parser.parse_args()
    _, protocol, rows = load_inputs()
    model_id, revision = CHECKPOINTS["3b-instruct"]
    # Match the published 3B screen and avoid rounding near-tied logits.
    device = "cpu"
    dtype = torch.float32
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype, attn_implementation="eager"
    ).eval().to(device)
    targets = [row for row in rows if row["split"] == "test" and row["task"] == "target"]
    demos = {}
    for frame in sorted({row["frame"] for row in targets}):
        groups = [[row for row in rows if row["task"] == "target" and row["split"] == "dev"
                   and row["gender"] == gender and row["frame"] == frame][:2]
                  for gender in ("common", "neuter")]
        assert all(len(group) == 2 for group in groups)
        demos[frame] = [row for pair in zip(*groups) for row in pair]
    prefixes = {frame: "\n".join(row["prompt"] + " " + row["expected"] + "." for row in examples) + "\n"
                for frame, examples in demos.items()}
    ids = [tokenizer.encode(" " + word, add_special_tokens=False) for word in ("min", "mitt")]
    assert all(len(sequence) == 1 for sequence in ids) and ids[0] != ids[1]
    min_id, mitt_id = (sequence[0] for sequence in ids)
    signs = torch.tensor([1 if row["expected"] == "min" else -1 for row in targets])
    scores = {}
    conditions = ("zero_shot", "four_shot")
    if args.compare_chat:
        conditions += ("chat_zero_shot", "chat_four_shot")
    for condition in conditions:
        prompts = [(prefixes[row["frame"]] if condition.endswith("four_shot") else "") + row["prompt"]
                   for row in targets]
        if condition.startswith("chat_"):
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": "Fyll i den sista meningen med min eller mitt."},
                        {"role": "assistant", "content": prompt},
                    ],
                    tokenize=False,
                    continue_final_message=True,
                )
                for prompt in prompts
            ]
            assert all(prompt.endswith(row["prompt"]) for prompt, row in zip(prompts, targets))
        for prompt in prompts:
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            for word, token in (("min", min_id), ("mitt", mitt_id)):
                assert tokenizer.encode(prompt + " " + word, add_special_tokens=False) == prefix + [token]
        margins = []
        for start in range(0, len(prompts), 8):
            batch = tokenizer(prompts[start:start + 8], add_special_tokens=False,
                              padding=True, return_tensors="pt").to(device)
            logits = model(**batch, use_cache=False).logits[:, -1].float().cpu()
            margins.append(logits[:, min_id] - logits[:, mitt_id])
        margins = torch.cat(margins)
        scores[condition] = correctness(margins, signs)
        for row, prompt, margin, score in zip(targets, prompts, margins, scores[condition]):
            row.update({condition + "_prompt": prompt, condition + "_margin": float(margin),
                        condition + "_correct": float(score)})
    counts = {}
    for condition, values in scores.items():
        groups = {gender: float(values[torch.tensor([row["gender"] == gender for row in targets])].mean())
                  for gender in ("common", "neuter")}
        counts[condition] = dict(accuracy=float(values.mean()), per_gender=groups,
                                 target_gate=all(score >= protocol["competence_threshold"] for score in groups.values()))
    delta = scores["four_shot"] - scores["zero_shot"]
    interval = paired_interval(delta, targets, torch.ones(len(targets), dtype=torch.bool),
                               protocol["bootstrap_samples"], protocol["seed"])
    folder = ROOT / "results" / ("prompting_chat" if args.compare_chat else "prompting")
    folder.mkdir(exist_ok=True, parents=True)
    summary = dict(model=model_id, revision=revision, dtype=str(dtype), device=device,
                   provenance="Exploratory reuse of the inspected follow-up test split.",
                   demonstrations=demos, conditions=counts, gain=float(delta.mean()), gain_ci95=interval,
                   fingerprints=hashes(["prompting_check.py", "checkpoint_check.py", "followup.py",
                                        "data/followup.json", "followup_protocol.json"]))
    if args.compare_chat:
        comparisons = (
            ("chat_zero_shot", "zero_shot"),
            ("chat_four_shot", "four_shot"),
            ("chat_four_shot", "chat_zero_shot"),
        )
        summary["bootstrap_seed"] = protocol["seed"]
        summary["bootstrap_samples"] = protocol["bootstrap_samples"]
        summary["comparisons"] = {}
        for treatment, reference in comparisons:
            difference = scores[treatment] - scores[reference]
            summary["comparisons"][f"{treatment}_vs_{reference}"] = dict(
                gain=float(difference.mean()),
                gain_ci95=paired_interval(difference, targets, torch.ones(len(targets), dtype=torch.bool),
                                          protocol["bootstrap_samples"], protocol["seed"]),
            )
    save_json(folder / "summary.json", summary)
    with (folder / "examples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)
    print(counts, "gain:", float(delta.mean()), "interval:", interval)


if __name__ == "__main__":
    main()
