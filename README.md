# Swedish agreement as a test of weight-space editing

Can the output-weight contrast between Swedish **den** and **det** improve
**min/mitt** possessive agreement on unseen nouns? This study tests a targeted
edit on Qwen2.5 and compares it with bias calibration, random edits, and few-shot
prompting.

[Research note](RESEARCH_NOTE.md) · [PDF report](ouputpdf/research_note.pdf)

## Results

- **Original experiment.** On 96 sentences covering 48 test nouns, the 0.5B Base
  model scored 50.0% before and after the fixed edit. Bias calibration scored
  49.0%. [Summary](results/summary.json) · [Per-example scores](results/examples.csv)
- **Checkpoint screen.** Two of four checkpoints met the 75%-per-gender
  threshold for **den/det**. All four fell short on **min/mitt**. The 0.5B Base
  probe ranked gender more accurately in source contexts (AUC 0.922) than in
  target contexts (AUC 0.604). [Checkpoint results](results/checkpoints/) ·
  [Follow-up summary](results/followup/summary.json) ·
  [Follow-up per-example scores](results/followup/examples.csv)
- **Few-shot prompting.** Four examples per frame raised 3B Instruct accuracy
  from 53.1% to 64.1% on the same test split. Both gender groups remained below
  the threshold. With a Swedish fill-in instruction and the chat template,
  accuracy rose from 39.1% zero-shot to 59.4% four-shot. Both genders still
  fell below the threshold. [Plain-completion summary](results/prompting/summary.json) ·
  [Per-example scores](results/prompting/examples.csv) ·
  [Chat-template comparison](results/prompting_chat/summary.json) ·
  [Chat-template per-example scores](results/prompting_chat/examples.csv)

Transfer remains unresolved. The benchmark measures forced-choice performance
on synthetic sentences with gender cues. These sentences still await review
by a Swedish speaker. The checkpoint screen, prompting
comparisons, and 3B strength sweep reused the same 32 follow-up test nouns.
Treat those results and their intervals as exploratory. The regression
measure tracks changes in expected-token log probability on 20 separate prefixes.

We recommend giving this particular weight edit low priority for Swedish model
development. It changed no choices in the original test, and a simple bias
adjustment matched or beat every development-selected edit. The wider research
question remains open. With more time, we would test the unedited model on
whole possessive phrases to see how much the sentence format explains.

## Reproduce the experiments

Use Python 3.13. From the repository root:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest test_followup.py
python experiment.py --validate-only
python followup.py --validate-only
~~~

Run the two experiments with:

~~~powershell
python experiment.py
python followup.py --output results/followup-reproduction
~~~

The scripts download pinned model weights on first use. Use `--device cpu`
with the experiment and checkpoint scripts to run on CPU. The prompting
script uses CPU by default. The original run replaces
`results/summary.json` and `results/examples.csv`.

Run the checkpoint and prompting comparisons with:

~~~powershell
python checkpoint_check.py 0.5b-base
python checkpoint_check.py 1.5b-base --device cpu
python checkpoint_check.py 1.5b-instruct
python checkpoint_check.py 3b-instruct --device cpu
python prompting_check.py
python prompting_check.py --compare-chat
~~~

### Model revisions and random seeds

| Checkpoint | Exact revision | Results |
| --- | --- | --- |
| Qwen2.5-0.5B Base | `060db6499f32faf8b98477b0a26969ef7d8b9987` | [Checkpoint summary](results/checkpoints/0.5b-base.json) |
| Qwen2.5-1.5B Base | `8faed761d45a263340a0528343f099c05c9a4323` | [Checkpoint summary](results/checkpoints/1.5b-base.json) |
| Qwen2.5-1.5B Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | [Checkpoint summary](results/checkpoints/1.5b-instruct.json) |
| Qwen2.5-3B Instruct | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | [Checkpoint summary](results/checkpoints/3b-instruct.json) |

The [original experiment](results/summary.json) uses seed **42** and random-edit
seeds **100–119**. The [follow-up](results/followup/summary.json) and checkpoint
screen use seed **314** and random-edit seeds **200–219**. The prompting
comparisons use **314** for the paired noun bootstrap.

The [original protocol](protocol.json) and [follow-up protocol](followup_protocol.json)
specify the edit strengths and evaluation criteria. The [benchmark data](data/)
include the development and test splits. The follow-up selects edit strength
and calibration bias on development nouns and estimates uncertainty by
resampling test nouns within gender, keeping both sentence frames together.

The [Swedish negation pilot](safety_study/RESULTS.md) found that both model
versions answered *Nej* to all 64 prompts. A [follow-up action-choice test](safety_study/DEVELOPMENT.md)
found identical responses before and after editing.

## License

Code is available under the [MIT license](LICENSE). Both synthetic benchmarks
are available under [CC0 1.0](data/LICENSE). The 0.5B Base, 1.5B Base, and 1.5B Instruct weights used here are distributed under Apache-2.0. The pinned 3B Instruct weights use the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE).
