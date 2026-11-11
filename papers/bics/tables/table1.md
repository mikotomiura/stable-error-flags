*Table 2. Difference in detection rate of stable errors between A and each control (q-mean over q = 0.1–0.5; 95% cluster-bootstrap interval). n: stable errors (independent units); n can be smaller than in Table 5 because items on which a primary method could not be parsed are excluded from the comparison (S2). Status: Section 2.4. Detection rates of each method, all-error results and other datasets are in S4. †: on WiC-type data OLMo's (ii) and (iii) predict 'different' on more than 99% of items and are not working judges (S8, P8); these differences are not evidence about the structure.*

| Study | Data | Classifier × LLM | Status | n (units) | A − (i) | A − (ii) | A − (iii) | A − (iv) |
|---|---|---|---|---|---|---|---|---|
| 1 | WiC test | DeBERTa × Qwen | H1/H2 cell; exploratory (check failed) | 208 (201) | 0.206 [0.142, 0.268] | 0.154 [0.096, 0.213] | 0.150 [0.088, 0.212] | 0.157 [0.087, 0.228] |
| 1 | WiC test | FLAN-T5 × Qwen | secondary | 293 (280) | 0.276 [0.221, 0.330] | 0.115 [0.070, 0.159] | 0.119 [0.072, 0.165] | 0.206 [0.145, 0.266] |
| 1 | WiC test | DeBERTa × OLMo | secondary | 206 (199) | 0.089 [0.026, 0.150] | 0.075 [0.017, 0.134] † | 0.055 [−0.004, 0.116] † | 0.067 [−0.002, 0.134] |
| 1 | ChaosNLI | RoBERTa × Qwen | H3 cell; exploratory (check failed) | 692 (665) | 0.041 [0.004, 0.078] | −0.097 [−0.119, −0.075] | −0.095 [−0.128, −0.059] | 0.079 [0.058, 0.100] |
| 1 | ChaosNLI | RoBERTa × OLMo | secondary | 680 (654) | 0.007 [−0.030, 0.044] | −0.110 [−0.141, −0.078] | −0.102 [−0.139, −0.067] | 0.076 [0.047, 0.104] |
| 2 | MCL-WiC dev+test | DeBERTa × Qwen | H1/H2 cell; aborted (91 < 100) | 91 (81) | 0.268 [0.161, 0.374] | 0.068 [−0.035, 0.175] | 0.044 [−0.058, 0.146] | 0.299 [0.191, 0.406] |
| 2 | MCL-WiC dev+test | FLAN-T5 × Qwen | H1b cell; aborted | 285 (225) | 0.350 [0.291, 0.410] | 0.076 [0.017, 0.135] | 0.093 [0.034, 0.151] | 0.156 [0.093, 0.221] |
| 2 | MCL-WiC dev+test | DeBERTa × OLMo | secondary | 91 (81) | 0.127 [0.018, 0.234] | 0.033 [−0.078, 0.152] † | 0.018 [−0.097, 0.138] † | 0.231 [0.137, 0.326] |
| 2 | MCL-WiC dev+test | FLAN-T5 × OLMo | secondary | 285 (225) | 0.221 [0.159, 0.283] | 0.133 [0.071, 0.194] † | 0.159 [0.100, 0.220] † | 0.036 [−0.023, 0.095] |
