*Table 3. How much of A's advantage over confidence an unstructured second judgement already reaches [P] (S8). Share = ((ii) − (i)) / (A − (i)) on the same stable errors (95% cluster-bootstrap interval; not bounded by 0 and 1). Debiased (ii): the same judge with its log-odds shifted by one constant per configuration so that it predicts 'different' as often as A does (no labels used). †: OLMo's (ii) predicts 'different' on more than 99% of items and is near chance as a judge (S8); these rows are not evidence about the structure.*

| Study | Data | Classifier × LLM | A − (i) | Share of (ii) | A − (ii) | Share of debiased (ii) | A − debiased (ii) |
|---|---|---|---|---|---|---|---|
| 1 | WiC test | DeBERTa × Qwen | 0.206 | 0.25 [−0.04, 0.49] | 0.154 [0.096, 0.213] | 0.39 [0.13, 0.63] | 0.125 [0.066, 0.185] |
| 1 | WiC test | FLAN-T5 × Qwen | 0.276 | 0.58 [0.43, 0.73] | 0.115 [0.070, 0.159] | 0.72 [0.57, 0.86] | 0.078 [0.036, 0.121] |
| 1 | WiC test | DeBERTa × OLMo † | 0.089 | 0.16 [−1.24, 0.75] | 0.075 [0.017, 0.134] | 0.38 [−0.58, 1.09] | 0.055 [−0.005, 0.117] |
| 2 | MCL-WiC dev+test | DeBERTa × Qwen | 0.268 | 0.75 [0.41, 1.17] | 0.068 [−0.035, 0.175] | 0.75 [0.44, 1.15] | 0.066 [−0.030, 0.167] |
| 2 | MCL-WiC dev+test | FLAN-T5 × Qwen | 0.350 | 0.78 [0.64, 0.95] | 0.076 [0.017, 0.135] | 0.88 [0.73, 1.05] | 0.042 [−0.017, 0.100] |
| 2 | MCL-WiC dev+test | DeBERTa × OLMo † | 0.127 | 0.74 [−0.28, 2.43] | 0.033 [−0.078, 0.152] | 0.41 [−1.00, 1.53] | 0.075 [−0.038, 0.193] |
| 2 | MCL-WiC dev+test | FLAN-T5 × OLMo † | 0.221 | 0.40 [0.20, 0.61] | 0.133 [0.071, 0.194] | 1.10 [0.81, 1.52] | −0.022 [−0.091, 0.048] |
