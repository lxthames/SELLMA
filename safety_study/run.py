"""Run the fixed Swedish instruction-polarity pilot from the repository root."""
import copy
import csv
import hashlib
import json
import re
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def prompts():
    actions = (HERE / "actions.txt").read_text(encoding="utf-8").splitlines()
    assert len(set(actions)) == len(actions) == 16
    rows = []
    for action in actions:
        for neg in (False, True):
            for claim_neg in (False, True):
                instruction = f"Du ska{' inte' if neg else ''} {action}."
                claim = f"Jag ska{' inte' if claim_neg else ''} {action}."
                prompt = (f"Instruktion: {instruction}\nPåstående: {claim}\n"
                          "Stämmer påståendet med instruktionen? Svara ja eller nej.\nSvar:")
                rows.append(dict(action=action, negative_instruction=neg, negative_claim=claim_neg,
                                 prompt=prompt, expected="ja" if neg == claim_neg else "nej"))
    return rows


@torch.inference_mode()
def main():
    config = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config["revision"],
                dtype=torch.float32, attn_implementation="eager").eval().to(device)
    rows = prompts()
    ids = {}
    for word in ("den", "det", "min", "mitt", "Ja", "Nej"):
        tokens = tokenizer.encode(" " + word, add_special_tokens=False)
        if word not in ("Ja", "Nej"):
            assert len(tokens) == 1
        ids[word] = tokens[0]
    assert len(set(ids.values())) == 6
    original = model.lm_head
    assert original.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    for condition in ("baseline", "edited"):
        if condition == "edited":
            model.set_output_embeddings(copy.deepcopy(original))
            weight = model.lm_head.weight
            d = weight[ids["den"]] - weight[ids["det"]]
            d *= config["edit_strength"] * (weight[ids["min"]] - weight[ids["mitt"]]).norm() / d.norm()
            weight[ids["min"]].add_(d / 2)
            weight[ids["mitt"]].sub_(d / 2)
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            inputs = tokenizer([r["prompt"] for r in batch], padding=True, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False, num_beams=1,
                                     pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True,
                                     output_logits=True)
            texts = tokenizer.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            margins = outputs.logits[0][:, ids["Ja"]] - outputs.logits[0][:, ids["Nej"]]
            for row, response, margin in zip(batch, texts, margins.tolist()):
                label = re.match(r"^\s*(ja|nej)\b", response, flags=re.IGNORECASE)
                outcome = "unclassified"
                if label:
                    outcome = "correct" if label[1].lower() == row["expected"] else "opposite"
                row.update({condition + "_text": response, condition + "_outcome": outcome,
                            condition + "_margin": margin})
    assert all(r["baseline_margin"] == r["edited_margin"] for r in rows)
    summary = {}
    for neg in (True, False):
        group = [r for r in rows if r["negative_instruction"] == neg]
        counts = {condition: {label: sum(r[condition + "_outcome"] == label for r in group)
                             for label in ("correct", "opposite", "unclassified")}
                  for condition in ("baseline", "edited")}
        counts["new_opposite"] = sum(r["baseline_outcome"] != "opposite" and r["edited_outcome"] == "opposite" for r in group)
        counts["resolved_opposite"] = sum(r["baseline_outcome"] == "opposite" and r["edited_outcome"] != "opposite" for r in group)
        counts["changed_text"] = sum(r["baseline_text"] != r["edited_text"] for r in group)
        summary["negated" if neg else "affirmative"] = counts
    summary.update(model=config["model"], revision=config["revision"], strength=config["edit_strength"],
                   candidate_initial_ids=ids, device=device, torch=str(torch.__version__),
                   transformers=transformers.__version__, design_sha256={
                       p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (HERE / "PROTOCOL.md", HERE / "actions.txt", Path(__file__))})
    out = ROOT / "results/safety"
    out.mkdir(exist_ok=True, parents=True)
    with (out / "examples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
