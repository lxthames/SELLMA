# Can a Swedish gender contrast transfer through a weight edit?

## Abstract

Can the den/det output-weight contrast improve Swedish possessive agreement for unseen nouns? Two of four Qwen2.5 checkpoints passed the source gate, and all four failed the possessive gate. Bias calibration matched or exceeded every edit selected on development data. Four-shot prompting lifted 3B Instruct accuracy from 53.1% to 64.1% on the reused test split, a gain of 10.9 percentage points (95% CI [1.6, 20.3]). Both genders remained below the 75% threshold. Transfer remains unresolved pending a speaker-reviewed competence test.

## Evidence from prior work

Experiment 3 gives the clearest evidence of reasoning from weights. GPT-6 Astra, the language model used to inspect and edit weights, examined four transformer blocks from each of three character models in Python. It could not run any layer on inputs. Its copying-head predictions matched the authors' independent measurements on repeated strings (Figure 1) [1].

In Experiment 7, four weights-only attempts failed to identify the model trained to agree with the user's suggested answer. Given attention patterns on prompts it wrote, Astra identified that model in all three Gemma pairs [1].

Experiment 8 allowed 50 batched practice forward passes. On Qwen3.5-0.8B, held-out MATH-500 accuracy rose from 7.9% to 48.1% as token-limit hits fell from 93.0% to 33.1%. On Gemma 3 270M, which rarely looped, the authors found no established math gain. Qwen's improvement is consistent with letting the model finish reasoning it could already perform. An edit search with the same feedback budget would help isolate the contribution of weight reasoning [1].

Swedish grammatical gender gives us a small transfer test because noun gender guides both anaphoric *den/det* and possessive *min/mitt*. We test whether the output-weight direction for the first contrast improves the second, using the baseline competence screen and control conditions to guide the recommendation.

## Experimental design

### Output-weight edit

Write $w_t$ for the output-weight row of token $t$. Define the unit source direction $u$ and the scaled edit $d_\alpha$ using the original distance between the possessive rows. The fixed edit used $\alpha=0.25$.

$$
u = \frac{w_{\mathrm{den}}-w_{\mathrm{det}}}{\lVert w_{\mathrm{den}}-w_{\mathrm{det}}\rVert_2},
\qquad
d_\alpha = \alpha\,\lVert w_{\mathrm{min}}-w_{\mathrm{mitt}}\rVert_2\,u .
$$

Apply equal and opposite updates to the two output rows. The input embeddings retain their original values.

$$
w'_{\mathrm{min}} = w_{\mathrm{min}} + \tfrac12 d_\alpha,
\qquad
w'_{\mathrm{mitt}} = w_{\mathrm{mitt}} - \tfrac12 d_\alpha .
$$

For a final hidden state $h$, let $m(h)$ denote the original *min* logit minus the *mitt* logit, and $m'(h)$ the edited margin. Their difference depends on the projection of the hidden state onto the edit direction.

$$
m'(h)-m(h) = h^\top d_\alpha .
$$

### Controls and evaluation

The original experiment compared baseline, the fixed edit, bias calibration, and 20 equal-norm random edits on 48 test nouns.

The follow-up used 16 development nouns to select strength and calibration bias. Articles and endings provided gender cues. Support required both gates, a positive gain interval, a gain above calibration and the random-edit 95th percentile, and mean regression of at least -0.05 nats.

For each of the 20 regression prefixes $x_i$, let $y_i$ be its expected next token. The regression measure $R_c$ averages the change in its log probability under condition $c$ relative to baseline $0$. A negative value denotes a loss.

$$
R_c=\frac{1}{20}\sum_{i=1}^{20}\left[\log p_c(y_i\mid x_i)-\log p_0(y_i\mid x_i)\right].
$$

## Baseline competence and prompting

Table 1. Baseline agreement by checkpoint and grammatical gender.

| Qwen2.5 | den/det (common / neuter) | Source gate | min/mitt (common / neuter) | Target gate |
| --- | ---: | --- | ---: | --- |
| 0.5B Base | 12.5% / 100.0% | Fail | 100.0% / 0.0% | Fail |
| 1.5B Base | 96.9% / 81.3% | Pass | 100.0% / 0.0% | Fail |
| 1.5B Instruct | 100.0% / 65.6% | Fail | 96.9% / 0.0% | Fail |
| 3B Instruct | 96.9% / 96.9% | Pass | 100.0% / 6.3% | Fail |

Each gate requires at least 75% accuracy for each gender. The four-checkpoint screen, prompting comparisons, and 3B strength sweep reused the same 32 previously inspected test nouns, with two frames per noun and task. These are exploratory diagnostics.

On 3B Instruct, four gender-balanced development examples per frame raised accuracy from 53.1% to 64.1%, a paired gain of 10.9 percentage points (95% CI [1.6, 20.3]) on the reused split. Prompting reduced the preference for min. Common accuracy fell to 56.3% and neuter accuracy rose to 71.9%, leaving both below the gate.

On 0.5B Base, “Jag har ett ankare. Jag använder” correctly favored *det* over *den* by 0.68 logits. “Ankaret är” favored *min* over the expected *mitt* by 2.47 logits at baseline and 2.81 after the fixed edit. On the regression prefix “Hon gjorde en sur”, calibration lowered the log probability of the expected *min* by 1.405 nats.

## Edit and calibration results

On the original 48 test nouns, 0.5B Base always chose min before and after the edit (50.0%). Calibration scored 49.0% and 20 equal-norm random edits scored 45.8% to 50.0%.

Table 2. Possessive accuracy and paired 95% gain intervals (percentage points). The second line in each treatment cell gives regression in nats.

| Checkpoint | Baseline | Fixed 0.25 | Selected edit | Bias calibration |
| --- | ---: | ---: | ---: | ---: |
| 0.5B Base | 50.0% | 50.0% [0.0, 0.0]<br>-0.015 | 50.0% [0.0, 0.0]<br>+0.000 | 54.7% [-4.7, 14.1]<br>+0.00045 |
| 1.5B Base | 50.0% | 50.0% [0.0, 0.0]<br>-0.178 | 50.0% [0.0, 0.0]<br>+0.000 | 50.0% [-7.8, 7.8]<br>+0.003 |
| 1.5B Instruct | 48.4% | 50.0% [0.0, 4.7]<br>-0.188 | 48.4% [0.0, 0.0]<br>+0.000 | 48.4% [0.0, 0.0]<br>+0.000 |
| 3B Instruct | 53.1% | 51.6% [-4.7, 0.0]<br>-0.103 | 54.7% [-3.1, 6.2]<br>-0.873 | 59.4% [-4.7, 17.2]<br>+0.007 |

No development-selected edit outscored bias calibration on the four-checkpoint screen. The fixed edit did score one more item than calibration on 1.5B Instruct (50.0% versus 48.4%), while missing every neuter item. The exploratory 3B sweep used development nouns to select strength 2.0 after earlier inspection of the test split. Its gain interval includes zero and its regression exceeded the allowed loss of 0.05 nats.

Development selected zero strength on 0.5B Base and both 1.5B checkpoints. Their selected edits and norm-matched random edits therefore reproduce baseline.

Calibration's near-zero means hide opposing shifts. On 0.5B, other uses of min lost 1.404 nats and spatial mitt gained 1.405. On 3B, the same groups lost 0.675 and gained 0.690 nats, respectively.

## Checkpoint and direction diagnostics

The screen covers the Qwen2.5 family [2]. GPT-SW3 356M is a candidate for a Swedish-trained comparison and requires approved access [3].

On 0.5B Base, projecting the final hidden state onto the den-minus-det direction ranked common prompts above neuter prompts with AUC 0.922 on source frames (95% CI 0.826–0.988) and 0.604 on target frames (0.517–0.680), counting ties as half. Thresholds fitted on the 16 development nouns scored 89.1% (71.9–95.3%) on source and 59.4% (48.4–65.6%) on target. The target accuracy interval includes chance.

Intervals use 10,000 bootstrap draws within gender, keeping both frames per noun together and pairing gains with baseline. Probe percentile intervals use seed 314, refitting thresholds on independently resampled development nouns for each test draw.

## Interpretation and recommendation

The checkpoint screen used plain completions to give base and Instruct models the same input and score the same next-token choice. A separate 3B Instruct check added a Swedish fill-in instruction in the chat template and prefilled the assistant reply with the same sentence and, for four-shot, the same demonstrations. On the reused test split, chat accuracy rose from 39.1% to 59.4% with demonstrations (gain 20.3 pp, 95% CI [6.2, 32.8]). Chat four-shot scored 46.9% on common nouns and 71.9% on neuter nouns, below the gate for both. The difference from plain four-shot was -4.7 pp (95% CI [-15.6, 6.2]).

Weak possessive agreement and a preference for min after “Flaskan är” could both explain the failed gate. The probe detects a gender signal, especially on den/det frames. The synthetic prompts still await independent Swedish speaker review. Whether an edit could unlock agreement remains unresolved.

Start with a checkpoint pretrained on Swedish text. If it still struggles with possessive agreement on a speaker-reviewed benchmark, fine-tune it on Swedish sentences that demonstrate common- and neuter-gender agreement, then measure gains on held-out nouns in both groups. This study leaves the effect of that training open. A targeted weight edit is worth testing when a checkpoint already demonstrates agreement and still makes a repeatable error.

On fresh, speaker-reviewed nouns, compare the current frame with full-phrase log likelihood. After a shared prefix, score “min cykel” against “mitt cykel” and “mitt hus” against “min hus”. Report each gender separately. Passing only the phrase test would implicate the frame. Keep plain and chat-formatted few-shot prompts as baselines because both improved with examples on the reused split. Once a checkpoint passes both gates, compare the weight edit and activation steering against those prompting baselines on untouched nouns, using the same regression checks.

## References

[1] Takehi et al., Sparks of Weight-Space Programming. See Experiment 3, Figure 1, Experiment 7, and the Qwen and Gemma tables under Experiment 8. https://rikiyat.github.io/blog/weight-space-programming/

[2] Qwen model family. https://huggingface.co/collections/Qwen/qwen25

[3] AI Sweden, GPT-SW3 356M access conditions. https://huggingface.co/AI-Sweden-Models/gpt-sw3-356m

## Appendix. Does the agreement edit affect Swedish negation?

The main regression measure tracks other uses of *min* and *mitt*. This pilot asks a separate question about instruction interpretation. Does the fixed agreement edit change how models respond when Swedish *inte* (“not”) changes what an instruction requires?

### Yes/no judgments of restatements

We gave Qwen2.5-0.5B Base 64 prompts about 16 fictional administrative actions. Each prompt paired an affirmative or negated instruction with a restatement that either matched or contradicted it. We scored the first word of the reply, *Ja* for a match and *Nej* for a contradiction, before and after the 0.25-strength edit.

For “Du ska inte spara kvittot” (“You should not keep the receipt”), a matching restatement is “Jag ska inte spara kvittot” (“I should not keep the receipt”), which calls for *Ja*. “Jag ska spara kvittot” (“I should keep the receipt”) calls for *Nej*. The model answered *Nej* in both cases and across all 64 prompts, giving 32 correct verdicts before and after editing. A constant answer makes this format uninformative about whether the model understood the negation.

### Choosing an action

We reused the 16 actions with the chat template of Qwen2.5-3B-Instruct and asked the model to choose between two concrete actions. Each scenario included an affirmative instruction, a negated instruction that reversed the correct choice, a paraphrase of that negation, and an instruction that negated a different action while leaving the correct choice unchanged. We presented both answer orders for each variant, giving 32 choices per variant.

Both model versions scored 28/32 on affirmative instructions and 32/32 on both the meaning-changing negation and its paraphrase. When *inte* applied to a different action, accuracy fell to 6/32. Swapping the order of the options changed the choice in eight of 64 pairs. The baseline and edited models gave identical responses throughout this check.

The action-choice format exposed errors that the yes/no format could not distinguish. Both the scope of negation and the order of the options affected responses in this development set. A held-out test with fresh, speaker-reviewed scenarios and questions focused on a single action would help establish whether the edit changes how models interpret negation.

## Model revisions and random seeds

Qwen2.5-0.5B Base — 060db6499f32faf8b98477b0a26969ef7d8b9987

Qwen2.5-1.5B Base — 8faed761d45a263340a0528343f099c05c9a4323

Qwen2.5-1.5B Instruct — 989aa7980e4cf806f80c7fef2b1adb7bc71aa306

Qwen2.5-3B Instruct — aa8e72537993ba99e69dfaafa59ed015b17504d1

The original experiment used seed 42 and random-edit seeds 100–119. The follow-up and checkpoint screen used seed 314 and random-edit seeds 200–219. Both prompting comparisons used seed 314 for paired noun resampling.

