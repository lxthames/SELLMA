# Can a Swedish gender contrast transfer through a weight edit?

## Abstract

On 48 held-out Swedish nouns, Qwen2.5-0.5B Base scored 50% before and after the fixed weight edit. It chose min every time. All four checkpoints fell short of our baseline criterion for possessive agreement: at least 75% accuracy for each gender. Whether this edit can transfer gender information in a model that already handles both constructions remains open. We recommend giving this edit low priority. Our next experiment would test the same fresh, speaker-reviewed nouns in sentence stems and complete possessive phrases.

## Hypothesis and experiment

Swedish noun gender guides both the pronoun pair *den/det* and the possessives *min/mitt*. We hypothesized that the difference between the model's output weights for *den* and *det* would help it choose the correct possessive for unseen nouns. This tests a practical idea for SE.LLMA: can a distinction the model uses in one construction correct an error in another?

In *Sparks of Weight-Space Programming*, the clearest evidence for reading weights directly came from copying-head predictions that the authors checked independently. The headline math edit allowed repeated model feedback, and fewer answers hit the token limit. That leaves open how much of the gain came from letting the model finish reasoning it could already perform. Our original experiment fixed the edit direction and strength before evaluation. [Original article](https://rikiyat.github.io/blog/weight-space-programming/)

We moved the two possessive output rows in opposite directions along the *den/det* contrast. We compared the fixed edit with the unchanged model, a bias adjustment fitted on development examples, and 20 random edits of equal size. Each of the 48 test nouns appeared in two sentence frames. We also checked whether editing reduced the probability of expected words on 20 separate prefixes.

## What the experiment showed

On the original test, the fixed edit changed none of the 96 possessive choices. Bias calibration scored 49.0%, and the random edits scored 45.8% to 50.0%. The failure is visible on a single follow-up example. After *Ankaret är* (the anchor is), the model preferred *min* over the correct *mitt*. The fixed edit increased that preference from 2.47 to 2.81 logits.

Two of four checkpoints met the baseline threshold for *den/det*. None met it for *min/mitt*. On 3B Instruct, the edit selected on development nouns scored 54.7%, compared with 53.1% at baseline and 59.4% for calibration. The selected edit also reduced expected-token log probability by 0.873 nats on average, exceeding the 0.05-nat loss limit. Four-shot prompting reached 64.1%. These later comparisons reused 32 previously inspected test nouns, so their gains are exploratory.

## Recommendation to SE.LLMA

Give this particular edit low priority for Swedish model development. It corrected no choices in the original test, and calibration matched or beat every development-selected edit. The broader transfer question remains open because the synthetic prompts still await independent speaker review and every checkpoint struggled with at least one gender on the possessive task.

With more time, we would test the unedited 3B model on fresh, speaker-reviewed nouns in both formats. For example, does it choose *mitt* after *Huset är*, and does it prefer the complete phrase *mitt hus* over *min hus*? We would report common and neuter nouns separately. If the model succeeds only on complete phrases, that would narrow the observed weakness to how we elicited agreement.

[Full PDF report](ouputpdf/research_note.pdf)
