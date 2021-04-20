# Results and Explanations

## Part 4: Position-Sensitive Attention

We tried several hyperparams for positional encodings. The basis for all of them is  \
`abs(relative_position) ** exponent`, with a cutoff value - positions further away than the cutoff value are considered to be at the cutoff value. For example, with `exponent=2, cutoff=3` the positional distance factors for the 2nd word are:
`-1 , 0 , -1 , -4 , -9 , -9 , ...`

We also tried normalization, since without normalization the positional distance factors have a large magnitude compared to the logits, and they dominate the attention weights. \
The techniques we tried:
* `None`: no normalization.
* `by logits magnitude`: linearly stretch the positional factors such that the largest positional factor is the same magnitude as the largest logit.
* `half cutoff = -1`: linearly stretch the positional factors such that the positional factor at *cutoff/2* equals -1.

| Normalization       | Exponent | Cutoff | Val Acc | Train Acc |
|---------------------|----------|--------|---------|-----------|
| None                | 1        | 10     | 0.580   | 0.715     |
| None                | 1.5      | 10     | 0.584   | 0.718     |
| None                | 2        | 10     | 0.585   | 0.717     |
| by logits magnitude | 1        | 10     | 0.573   | 0.711     |
| by logits magnitude | 1.5      | 10     | 0.575   | 0.717     |
| by logits magnitude | 2        | 10     | 0.573   | 0.715     |
| half cutoff = -1    | 1        | 10     | 0.561   | 0.702     |
| half cutoff = -1    | 1.5      | 10     |         |           |
| half cutoff = -1    | 2        | 10     |         |           |

Disappointingly, we got the best results when we didn't use any normalization, which suggests that closer words are usually much more relevant than further words. The exponent didn't have a strong impact on the results, with larger exponents usually having slightly better results.

## Part 5: Causal Attention

For word `i` we can set the positional factors of all the words `j>i` to be `-inf` (or actually `-1e12`), which effectively sets their attention weights to 0, making the attention causal since words cannot attend to words that appear later in the sequence.

Overall, the performance of the causal models was worse than the unhindered models, probably since valuable information can be found later in the sentence, e.g. in the sentence **She used her axe to deliver the final blow**, the word **axe** is very relevant to the word **used**.

| Normalization       | Exponent | Cutoff | Val Acc | Train Acc |
|---------------------|----------|--------|---------|-----------|
| None                | 1        | 10     | 0.565   | 0.684     |
| None                | 1.5      | 10     | 0.567   | 0.684     |
| None                | 2        | 10     | 0.566   | 0.683     |
| by logits magnitude | 1        | 10     | 0.564   | 0.695     |
| by logits magnitude | 1.5      | 10     | 0.566   | 0.703     |
| by logits magnitude | 2        | 10     | 0.565   | 0.705     |
| half cutoff = -1    | 1        | 10     |         |           |
| half cutoff = -1    | 1.5      | 10     |         |           |
| half cutoff = -1    | 2        | 10     |         |           |
