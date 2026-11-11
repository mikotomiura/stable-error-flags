*Table 4. Human adjudication of DeBERTa's stable errors [P] (two raters independent of the study, blind to the reference, the model and the source dataset; S9). Confirmed: both raters chose the reference label. Contested: all other errors; the next column counts those where both raters chose the opposite label. Counts are those of each comparison (OLMo excludes its parse failures). Baseline: on the stable-correct items mixed into the same sheet, both raters chose the reference label on 180 of 208 (Study 1) and 89 of 91 (Study 2). Study 2's contested class has 15 errors in 14 units; its intervals are not interpretable. †: OLMo's (ii) is not a working judge (Table 2); its rows are not evidence about the structure. Debiased (ii): as in Table 3.*

| Study | LLM | Stable errors | Confirmed | Contested | Both raters opposite | A − (ii), confirmed | A − (ii), contested | A − debiased (ii), confirmed | A − debiased (ii), contested |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen | 208 | 100 | 108 | 68 | 0.188 [0.104, 0.274] | 0.122 [0.044, 0.202] | 0.156 [0.070, 0.244] | 0.096 [0.017, 0.177] |
| 1 | OLMo † | 206 | 100 | 106 | 66 | 0.060 [−0.024, 0.146] | 0.089 [0.011, 0.166] | — | — |
| 2 | Qwen | 91 | 76 | 15 | 8 | 0.063 [−0.041, 0.173] | 0.093 [−0.229, 0.387] | 0.061 [−0.040, 0.166] | 0.093 [−0.200, 0.360] |
| 2 | OLMo † | 91 | 76 | 15 | 8 | −0.005 [−0.131, 0.124] | 0.227 [−0.038, 0.507] | — | — |
