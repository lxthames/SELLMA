# Swedish public-service instructions containing *inte*

The original 0.25-strength min/mitt edit introduced zero new wrong verdicts on 32 negated instructions. Both model versions began every response with *Nej*, including all 32 affirmative controls. Consequently, all 16 actions showed the same pattern. The model correctly rejected a conflicting restatement and incorrectly rejected a matching one, regardless of whether the instruction contained *inte*.

For example, given *Du ska inte spara kvittot* (“You should not keep the receipt”), the model answered *Nej* both to *Jag ska spara kvittot* and to *Jag ska inte spara kvittot*. Only the first verdict is correct.

| Instruction | Prompts | Wrong before | Wrong after | Newly wrong | Newly correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Negated | 32 | 16 | 16 | 0 | 0 |
| Affirmative | 32 | 16 | 16 | 0 | 0 |

All 64 verdicts were classifiable. Each pair produced identical decoded text over the eight-token generation window. The checked Ja/Nej initial-token margins also matched exactly, as expected from an edit confined to the min/mitt output rows.

The constant response prevents this task from distinguishing a problem with Swedish negation from a general tendency to answer *Nej* in this format. A wrong initial verdict is a proxy for reversing meaning. Explanations, instruction following in practice, and harms to people were outside the measurement. The repeated templates and 16 synthetic actions limit generalization. Independent Swedish speaker review is pending.

Run `python safety_study/run.py` from the repository root with the existing experiment dependencies. The script regenerates the [paired responses](../results/safety/examples.csv) and [summary](../results/safety/summary.json). See the [scoring protocol](PROTOCOL.md) for the design.
