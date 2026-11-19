# Negation contrast-set development check

Use the cached Qwen2.5-3B-Instruct checkpoint at revision aa8e72537993ba99e69dfaafa59ed015b17504d1, with its chat template and greedy decoding. Use float16 on CUDA and float32 on CPU. Compare its baseline with the original 0.25-strength output edit. The 16 original actions are development material. Each action gets an affirmative instruction, a meaning-changing negation, a paraphrase of the negation, and an irrelevant negation of a second action. Present every case with both orders of the concrete action options. Save generated answers as well as scores.

The development screen requires at least 75% of scenarios to get both the affirmative and negated cases right in both option orders. Also report the fraction getting all four variants right in both orders, order disagreement among classifiable pairs, and unclassifiable answers. This is an exploratory format check with a chosen threshold. Sixteen free-text responses per model condition accompany the scores for manual inspection.

For a later held-out comparison, have two Swedish speakers independently annotate fresh scenarios and resolve differences before model inference. Freeze the instructions, edit, format, and decoding. Report newly wrong and newly correct responses, and cluster the paired interval by scenario to keep variants together. Agree on the maximum acceptable increase before collecting results. Independent speaker review and held-out inference are pending.

The variant design draws on [CondaQA](https://aclanthology.org/2022.emnlp-main.598/) and [ScoNe](https://aclanthology.org/2023.acl-short.154/). The two orders follow the concern documented in [option-order sensitivity research](https://aclanthology.org/2024.findings-naacl.130/).

## Observed results

Both versions of the same checkpoint passed the simple contrast screen on 12 of 16 scenarios. Only one scenario was correct across all four variants and both option orders. All 128 choice responses were classifiable. Eight of the 64 option swaps changed the semantic choice.

| Variant | Baseline correct | Edited correct |
| --- | ---: | ---: |
| Affirmative | 28/32 | 28/32 |
| Negated | 32/32 | 32/32 |
| Negation paraphrase | 32/32 | 32/32 |
| Unrelated negation | 6/32 | 6/32 |

All 144 decoded responses matched before and after the edit. The free-text answers show why the forced-choice results need caution. For example, the model reframed a prohibition on taking a queue ticket as a restriction on using literature or other resources. These outputs are available for review and carry no independent speaker scores. The unrelated-negation prompt uses a broad question about what to do. Before a held-out run, narrow the question to the action being scored and ask reviewers to check that the second instruction is clearly unrelated.

Run `python safety_study/check_format.py` to reproduce the [generated responses](../results/safety_development/examples.csv) and [score summary](../results/safety_development/summary.json).
