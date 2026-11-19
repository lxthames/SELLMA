"""Compare pinned Qwen checkpoints on the already inspected follow-up benchmark."""
import argparse
from pathlib import Path

import torch

from followup import ROOT, collect, evaluate, hashes, load_inputs, save_json, select_settings


CHECKPOINTS = {
    "0.5b-base": ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
    "1.5b-base": ("Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323"),
    "1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "3b-instruct": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
}


def direction_probe(cache, rows):
    # Positive means a stronger common-gender preference. Scaling by a positive
    # constant preserves rank and the zero threshold.
    score = cache["hidden"] @ (cache["direction"] / cache["direction"].norm())
    result = {}
    for task in ("source", "target"):
        def values(split, gender):
            return score[[i for i, row in enumerate(rows)
                          if (row["task"], row["split"], row["gender"]) == (task, split, gender)]]

        common_dev, neuter_dev = (values("dev", gender) for gender in ("common", "neuter"))
        unique = torch.cat((common_dev, neuter_dev)).unique(sorted=True)
        thresholds = [float("-inf"), *(((unique[1:] + unique[:-1]) / 2).tolist()), float("inf"), 0.0]

        def accuracy(common, neuter, threshold):
            common_acc = (common > threshold).float() + 0.5 * (common == threshold).float()
            neuter_acc = (neuter < threshold).float() + 0.5 * (neuter == threshold).float()
            return ((common_acc.mean() + neuter_acc.mean()) / 2).item()

        threshold = min(thresholds, key=lambda x: (-accuracy(common_dev, neuter_dev, x), abs(x), x))
        common, neuter = (values("test", gender) for gender in ("common", "neuter"))
        pairs = common[:, None] - neuter
        result[task] = dict(
            auc=((pairs > 0).float() + 0.5 * (pairs == 0).float()).mean().item(),
            common_mean=common.mean().item(), neuter_mean=neuter.mean().item(),
            zero_threshold_accuracy=accuracy(common, neuter, 0.0),
            fitted_threshold=threshold if abs(threshold) != float("inf") else str(threshold),
            fitted_threshold_accuracy=accuracy(common, neuter, threshold),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", choices=CHECKPOINTS)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    _, protocol, rows = load_inputs()
    protocol["model"], protocol["revision"] = CHECKPOINTS[args.checkpoint]
    torch.manual_seed(protocol["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    cache = collect(protocol, rows, args.device)
    ids = cache["ids"]
    margin = cache["logits"][:, ids["min"]] - cache["logits"][:, ids["mitt"]]
    signs = torch.tensor([1 if row["expected"] == "min" else -1 for row in rows])
    settings = select_settings(margin, cache["hidden"] @ cache["direction"], signs, rows, protocol)
    result, _ = evaluate(cache, protocol, rows, settings)
    result["direction_probe"] = direction_probe(cache, rows)
    result["fingerprints"] = hashes(["checkpoint_check.py", "followup.py", "data/followup.json",
                                    "data/benchmark.json", "followup_protocol.json"])
    folder = ROOT / "results/checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / f"{args.checkpoint}.json", result)
    print(args.checkpoint, result["competence"], result["direction_probe"], result["support_checks"])


if __name__ == "__main__":
    main()
