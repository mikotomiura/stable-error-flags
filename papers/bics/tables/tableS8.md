*Table S8.1. Study 1 − Study 2 difference in A − control, with an independent cluster bootstrap per study [P].*

| Classifier × LLM | A − (i) | A − (ii) | A − (iii) |
|---|---|---|---|
| DeBERTa × Qwen | −0.062 [−0.185, 0.059] | 0.086 [−0.032, 0.203] | 0.106 [−0.011, 0.222] |
| FLAN-T5 × Qwen | −0.074 [−0.155, 0.006] | 0.039 [−0.034, 0.111] | 0.025 [−0.051, 0.101] |
| DeBERTa × OLMo † | −0.038 [−0.161, 0.082] | 0.042 [−0.092, 0.172] | 0.038 [−0.095, 0.168] |

*Table S8.2. Detection rate of stable errors by the classifier's prediction ŷ (same / different), and the share of all ranked items each method predicts as different [P].*

| Study | Classifier × LLM | ŷ | n | A | (i) | (ii) | (iii) | A − (ii) |
|---|---|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | same | 115 | 0.553 | 0.327 | 0.522 | 0.563 | 0.031 |
| 1 | DeBERTa × Qwen | different | 93 | 0.454 | 0.273 | 0.148 | 0.105 | 0.305 |
| 1 | FLAN-T5 × Qwen | same | 84 | 0.488 | 0.245 | 0.598 | 0.662 | −0.110 |
| 1 | FLAN-T5 × Qwen | different | 209 | 0.453 | 0.164 | 0.247 | 0.216 | 0.206 |
| 1 | DeBERTa × OLMo † | same | 115 | 0.466 | 0.322 | 0.567 | 0.602 | −0.101 |
| 1 | DeBERTa × OLMo † | different | 91 | 0.297 | 0.277 | 0.000 | 0.000 | 0.297 |
| 2 | DeBERTa × Qwen | same | 73 | 0.616 | 0.356 | 0.663 | 0.660 | −0.047 |
| 2 | DeBERTa × Qwen | different | 18 | 0.778 | 0.478 | 0.244 | 0.378 | 0.533 |
| 2 | FLAN-T5 × Qwen | same | 14 | 0.357 | 0.429 | 0.643 | 0.771 | −0.286 |
| 2 | FLAN-T5 × Qwen | different | 271 | 0.567 | 0.195 | 0.472 | 0.447 | 0.095 |
| 2 | DeBERTa × OLMo † | same | 73 | 0.493 | 0.356 | 0.592 | 0.611 | −0.099 |
| 2 | DeBERTa × OLMo † | different | 18 | 0.567 | 0.478 | 0.000 | 0.000 | 0.567 |
| 2 | FLAN-T5 × OLMo † | same | 14 | 0.386 | 0.429 | 0.929 | 0.957 | −0.543 |
| 2 | FLAN-T5 × OLMo † | different | 271 | 0.430 | 0.195 | 0.262 | 0.232 | 0.168 |

| Study | Classifier × LLM | Predicted different: A | (ii) | (iii) | classifier | reference |
|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | 0.571 | 0.712 | 0.804 | 0.480 | 0.537 |
| 1 | FLAN-T5 × Qwen | 0.571 | 0.712 | 0.804 | 0.653 | 0.537 |
| 1 | DeBERTa × OLMo † | 0.624 | 0.994 | 0.999 | 0.473 | 0.534 |
| 2 | DeBERTa × Qwen | 0.503 | 0.630 | 0.686 | 0.398 | 0.499 |
| 2 | FLAN-T5 × Qwen | 0.503 | 0.630 | 0.686 | 0.819 | 0.499 |
| 2 | DeBERTa × OLMo † | 0.523 | 0.999 | 0.998 | 0.398 | 0.499 |
| 2 | FLAN-T5 × OLMo † | 0.523 | 0.999 | 0.998 | 0.819 | 0.499 |

*Table S8.3. WiC test split by whether both sentences are WordNet usage examples (760 of 1,400 items) [P].*

| Classifier × LLM | Part | n (units) | A − (i) | A − (ii) | A − (iii) |
|---|---|---|---|---|---|
| DeBERTa × Qwen | both WordNet examples | 107 (105) | 0.250 [0.171, 0.329] | 0.127 [0.045, 0.209] | 0.133 [0.043, 0.221] |
| DeBERTa × Qwen | other | 101 (98) | 0.158 [0.062, 0.256] | 0.182 [0.101, 0.265] | 0.168 [0.088, 0.250] |
| FLAN-T5 × Qwen | both WordNet examples | 144 (142) | 0.311 [0.234, 0.385] | 0.111 [0.046, 0.177] | 0.108 [0.037, 0.179] |
| FLAN-T5 × Qwen | other | 149 (145) | 0.242 [0.166, 0.317] | 0.119 [0.057, 0.184] | 0.129 [0.065, 0.192] |
| DeBERTa × OLMo † | both WordNet examples | 106 (104) | 0.155 [0.074, 0.234] | 0.047 [−0.046, 0.141] | 0.021 [−0.071, 0.113] |
| DeBERTa × OLMo † | other | 100 (97) | 0.020 [−0.071, 0.111] | 0.104 [0.036, 0.171] | 0.092 [0.016, 0.164] |

*Table S8.4. Ceiling-normalised detection [P]: (detection − 0.30) / (ceiling − 0.30), where 0.30 is the q-mean of a random ranking and the ceiling is the q-mean of min(1, qN/n) for N ranked items and n stable errors. A − control on the same scale.*

| Study | Classifier × LLM | N | n (share) | Ceiling | A | (i) | (ii) | A − (i) | A − (ii) | A − debiased (ii) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | 1,304 | 208 (0.160) | 0.925 | 0.334 | 0.005 | 0.088 | 0.329 [0.227, 0.429] | 0.246 [0.153, 0.340] | 0.200 [0.106, 0.295] |
| 1 | FLAN-T5 × Qwen | 1,304 | 293 (0.225) | 0.867 | 0.287 | −0.199 | 0.084 | 0.486 [0.390, 0.582] | 0.203 [0.124, 0.281] | 0.138 [0.063, 0.213] |
| 1 | DeBERTa × OLMo † | 1,270 | 206 (0.162) | 0.923 | 0.146 | 0.003 | 0.026 | 0.143 [0.042, 0.241] | 0.120 [0.027, 0.216] | 0.089 [−0.008, 0.188] |
| 2 | DeBERTa × Qwen | 1,653 | 91 (0.055) | 1.000 | 0.498 | 0.115 | 0.400 | 0.383 [0.230, 0.534] | 0.097 [−0.050, 0.250] | 0.094 [−0.043, 0.239] |
| 2 | FLAN-T5 × Qwen | 1,653 | 285 (0.172) | 0.916 | 0.416 | −0.152 | 0.292 | 0.568 [0.472, 0.665] | 0.124 [0.027, 0.219] | 0.068 [−0.027, 0.162] |
| 2 | DeBERTa × OLMo † | 1,653 | 91 (0.055) | 1.000 | 0.297 | 0.115 | 0.250 | 0.182 [0.026, 0.334] | 0.047 [−0.112, 0.217] | 0.107 [−0.054, 0.276] |
| 2 | FLAN-T5 × OLMo † | 1,653 | 285 (0.172) | 0.916 | 0.207 | −0.152 | −0.009 | 0.359 [0.258, 0.460] | 0.215 [0.115, 0.315] | −0.035 [−0.147, 0.077] |

| Classifier × LLM | Study 1 − Study 2, normalised A − (ii) | normalised A − debiased (ii) |
|---|---|---|
| DeBERTa × Qwen | 0.149 [−0.026, 0.321] | 0.106 [−0.065, 0.271] |
| FLAN-T5 × Qwen | 0.079 [−0.043, 0.201] | 0.070 [−0.050, 0.189] |
| DeBERTa × OLMo † | 0.073 [−0.123, 0.264] | −0.018 [−0.207, 0.176] |

*Table S8.5. The LLM judges as classifiers over all ranked items [P]: accuracy against the reference, share predicted 'different', share of items whose score is tied with another item, and the largest share of tied items at a flag boundary. (i): the classifier's confidence (ties only). †: OLMo's (ii) and (iii) are not working judges (Table 2).*

| Study | LLM | Judge | Accuracy | Predicted different | Tied scores | Ties at boundary (max) |
|---|---|---|---|---|---|---|
| 1 | Qwen | (A) | 0.742 | 0.571 | 0.017 | 0.0008 |
| 1 | Qwen | (ii) | 0.668 | 0.712 | 0.008 | 0.0008 |
| 1 | Qwen | (iii) | 0.634 | 0.804 | 0.438 | 0.0008 |
| 1 | Qwen | (iv) | 0.546 | 0.982 | 0.751 | 0.3911 |
| 1 | — | (i) | — | — | 0.000 | 0.0008 |
| 1 | OLMo | (A) | 0.634 | 0.624 | 0.000 | 0.0008 |
| 1 | OLMo | (ii) | 0.537 | 0.994 | 0.000 | 0.0008 |
| 1 | OLMo | (iii) | 0.533 | 0.999 | 0.000 | 0.0008 |
| 1 | OLMo | (iv) | 0.523 | 0.928 | 0.000 | 0.0008 |
| 2 | Qwen | (A) | 0.821 | 0.503 | 0.002 | 0.0006 |
| 2 | Qwen | (ii) | 0.777 | 0.630 | 0.005 | 0.0006 |
| 2 | Qwen | (iii) | 0.748 | 0.686 | 0.313 | 0.0006 |
| 2 | Qwen | (iv) | 0.566 | 0.435 | 0.000 | 0.0006 |
| 2 | — | (i) | — | — | 0.000 | 0.0006 |
| 2 | OLMo | (A) | 0.675 | 0.523 | 0.000 | 0.0006 |
| 2 | OLMo | (ii) | 0.500 | 0.999 | 0.000 | 0.0006 |
| 2 | OLMo | (iii) | 0.502 | 0.998 | 0.000 | 0.0006 |
| 2 | OLMo | (iv) | 0.534 | 0.397 | 0.000 | 0.0006 |

*Table S8.6. Combining ranks with confidence (mean of ranks, as the frozen (i)+A) on stable errors, and AURC over abstention rates 0–0.95 on all items with a reference [P].*

| Study | Classifier × LLM | (i)+(ii) − (i) | (i)+A − (i)+(ii) | (i)+A − A | AURC (i) | AURC A | AURC (i)+A | AURC (ii) | AURC (i)+(ii) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | 0.023 [−0.014, 0.059] | 0.106 [0.067, 0.147] | −0.076 [−0.116, −0.037] | 0.191 | 0.144 | 0.145 | 0.179 | 0.172 |
| 1 | FLAN-T5 × Qwen | 0.063 [0.036, 0.091] | 0.067 [0.039, 0.095] | −0.146 [−0.186, −0.104] | 0.434 | 0.237 | 0.292 | 0.278 | 0.329 |
| 1 | DeBERTa × OLMo † | 0.012 [−0.030, 0.052] | 0.057 [0.016, 0.098] | −0.020 [−0.056, 0.016] | 0.193 | 0.183 | 0.171 | 0.222 | 0.198 |
| 2 | DeBERTa × Qwen | 0.155 [0.098, 0.209] | 0.048 [−0.022, 0.115] | −0.065 [−0.131, 0.000] | 0.115 | 0.065 | 0.066 | 0.092 | 0.085 |
| 2 | FLAN-T5 × Qwen | 0.142 [0.113, 0.172] | 0.048 [0.014, 0.081] | −0.160 [−0.206, −0.115] | 0.421 | 0.182 | 0.237 | 0.217 | 0.273 |
| 2 | DeBERTa × OLMo † | 0.114 [0.045, 0.180] | 0.022 [−0.066, 0.112] | 0.009 [−0.056, 0.074] | 0.115 | 0.121 | 0.093 | 0.131 | 0.100 |
| 2 | FLAN-T5 × OLMo † | 0.045 [0.019, 0.070] | 0.076 [0.039, 0.114] | −0.100 [−0.144, −0.058] | 0.421 | 0.273 | 0.302 | 0.340 | 0.361 |

*Table S8.7. A − (ii) and A − debiased (ii) within the classifier's prediction ŷ, with cluster-bootstrap intervals within the stratum [P].*

| Study | Classifier × LLM | ŷ | n (units) | A − (ii) | A − debiased (ii) |
|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | same | 115 (114) | 0.031 [−0.040, 0.104] | 0.113 [0.042, 0.186] |
| 1 | DeBERTa × Qwen | different | 93 (89) | 0.305 [0.220, 0.391] | 0.140 [0.042, 0.237] |
| 1 | FLAN-T5 × Qwen | same | 84 (81) | −0.110 [−0.176, −0.043] | 0.050 [−0.017, 0.117] |
| 1 | FLAN-T5 × Qwen | different | 209 (199) | 0.206 [0.154, 0.257] | 0.090 [0.036, 0.144] |
| 1 | DeBERTa × OLMo † | same | 115 (114) | −0.101 [−0.175, −0.028] | 0.064 [−0.017, 0.144] |
| 1 | DeBERTa × OLMo † | different | 91 (87) | 0.297 [0.229, 0.366] | 0.044 [−0.048, 0.136] |
| 2 | DeBERTa × Qwen | same | 73 (63) | −0.047 [−0.153, 0.061] | −0.005 [−0.111, 0.101] |
| 2 | DeBERTa × Qwen | different | 18 (18) | 0.533 [0.322, 0.700] | 0.356 [0.133, 0.556] |
| 2 | FLAN-T5 × Qwen | same | 14 (13) | −0.286 [−0.462, −0.086] | 0.014 [−0.160, 0.215] |
| 2 | FLAN-T5 × Qwen | different | 271 (212) | 0.095 [0.035, 0.153] | 0.044 [−0.017, 0.102] |
| 2 | DeBERTa × OLMo † | same | 73 (63) | −0.099 [−0.213, 0.020] | 0.058 [−0.070, 0.189] |
| 2 | DeBERTa × OLMo † | different | 18 (18) | 0.567 [0.389, 0.733] | 0.144 [−0.133, 0.400] |
| 2 | FLAN-T5 × OLMo † | same | 14 (13) | −0.543 [−0.662, −0.415] | 0.029 [−0.231, 0.262] |
| 2 | FLAN-T5 × OLMo † | different | 271 (212) | 0.168 [0.105, 0.229] | −0.024 [−0.098, 0.048] |

*Table S8.8. Sensitivity of the debiased controls to the target share of 'different' (A's share as in Table 3, 0.5, the classifier's own share), and stratified flags (top q within each predicted-label stratum) [P].*

| Study | Classifier × LLM | Target: A | 0.5 | classifier | A − debiased (iii), target A | A − (ii), stratified flags | A − (iii), stratified flags |
|---|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | 0.125 [0.066, 0.185] | 0.125 [0.062, 0.189] | 0.125 [0.062, 0.189] | 0.125 [0.062, 0.189] | 0.122 [0.062, 0.183] | 0.122 [0.058, 0.186] |
| 1 | FLAN-T5 × Qwen | 0.078 [0.036, 0.121] | 0.061 [0.016, 0.105] | 0.104 [0.060, 0.148] | 0.061 [0.017, 0.106] | 0.078 [0.034, 0.122] | 0.076 [0.033, 0.119] |
| 1 | DeBERTa × OLMo † | 0.055 [−0.005, 0.117] | 0.051 [−0.011, 0.115] | 0.050 [−0.013, 0.115] | 0.038 [−0.022, 0.099] | 0.071 [0.010, 0.133] | 0.065 [0.003, 0.129] |
| 2 | DeBERTa × Qwen | 0.066 [−0.030, 0.167] | 0.073 [−0.022, 0.172] | 0.138 [0.043, 0.236] | 0.090 [−0.011, 0.191] | 0.097 [0.000, 0.196] | 0.086 [−0.015, 0.185] |
| 2 | FLAN-T5 × Qwen | 0.042 [−0.017, 0.100] | 0.041 [−0.018, 0.099] | 0.166 [0.105, 0.225] | 0.013 [−0.043, 0.069] | 0.040 [−0.019, 0.098] | 0.036 [−0.021, 0.093] |
| 2 | DeBERTa × OLMo † | 0.075 [−0.038, 0.193] | 0.079 [−0.033, 0.195] | 0.130 [0.024, 0.240] | 0.048 [−0.067, 0.166] | 0.097 [−0.011, 0.211] | 0.077 [−0.034, 0.191] |
| 2 | FLAN-T5 × OLMo † | −0.022 [−0.091, 0.048] | −0.025 [−0.094, 0.044] | 0.075 [0.008, 0.142] | 0.013 [−0.055, 0.081] | −0.022 [−0.092, 0.047] | 0.011 [−0.057, 0.079] |

*Table S8.9. Using confidence and A together at the same flag budget [P]: union (flags taken alternately from A's and (i)'s rankings) and mean of ranks, against A alone (detection of stable errors).*

| Study | Classifier × LLM | A | (i) | Union | Mean of ranks | Union − A | Union − (i) |
|---|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | 0.509 | 0.303 | 0.414 | 0.432 | −0.094 [−0.127, −0.059] | 0.112 [0.064, 0.161] |
| 1 | FLAN-T5 × Qwen | 0.463 | 0.187 | 0.332 | 0.317 | −0.130 [−0.159, −0.100] | 0.145 [0.103, 0.188] |
| 1 | DeBERTa × OLMo | 0.391 | 0.302 | 0.326 | 0.371 | −0.065 [−0.102, −0.026] | 0.024 [−0.016, 0.066] |
| 2 | DeBERTa × Qwen | 0.648 | 0.380 | 0.574 | 0.583 | −0.075 [−0.117, −0.030] | 0.193 [0.105, 0.286] |
| 2 | FLAN-T5 × Qwen | 0.556 | 0.206 | 0.393 | 0.396 | −0.164 [−0.192, −0.133] | 0.187 [0.136, 0.238] |
| 2 | DeBERTa × OLMo | 0.508 | 0.380 | 0.448 | 0.516 | −0.059 [−0.115, 0.002] | 0.068 [−0.008, 0.147] |
| 2 | FLAN-T5 × OLMo | 0.427 | 0.206 | 0.305 | 0.327 | −0.123 [−0.152, −0.091] | 0.098 [0.049, 0.150] |

*Table S8.10. AUROC of each score for stable errors against all other ranked items [P], and the Study 1 − Study 2 difference of A − (ii) on this scale (independent bootstraps per study).*

| Study | Classifier × LLM | A | (i) | (ii) | (iii) | A − (ii) |
|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | 0.690 | 0.537 | 0.578 | 0.581 | 0.112 [0.073, 0.151] |
| 1 | FLAN-T5 × Qwen | 0.688 | 0.376 | 0.578 | 0.578 | 0.110 [0.079, 0.140] |
| 1 | DeBERTa × OLMo † | 0.623 | 0.534 | 0.536 | 0.546 | 0.087 [0.045, 0.129] |
| 2 | DeBERTa × Qwen | 0.767 | 0.579 | 0.713 | 0.748 | 0.054 [−0.019, 0.131] |
| 2 | FLAN-T5 × Qwen | 0.765 | 0.430 | 0.696 | 0.681 | 0.069 [0.036, 0.103] |
| 2 | DeBERTa × OLMo † | 0.672 | 0.579 | 0.661 | 0.668 | 0.011 [−0.069, 0.094] |
| 2 | FLAN-T5 × OLMo † | 0.662 | 0.430 | 0.538 | 0.512 | 0.124 [0.084, 0.163] |

| Classifier × LLM | Study 1 − Study 2, AUROC A − (ii) |
|---|---|
| DeBERTa × Qwen | 0.058 [−0.028, 0.140] |
| FLAN-T5 × Qwen | 0.041 [−0.005, 0.085] |
| DeBERTa × OLMo | 0.076 [−0.018, 0.167] |

*Table S8.11. ChaosNLI: the LLM judges as classifiers (accuracy; share of each predicted label) and AURC over abstention rates 0–0.95 [P].*

| LLM | Judge | Accuracy | entailment | neutral | contradiction |
|---|---|---|---|---|---|
| Qwen | (A) | 0.520 | 0.05 | 0.88 | 0.06 |
| Qwen | (ii) | 0.665 | 0.19 | 0.70 | 0.10 |
| Qwen | (iii) | 0.703 | 0.32 | 0.46 | 0.22 |
| Qwen | reference | — | 0.37 | 0.45 | 0.18 |
| OLMo | (A) | 0.530 | 0.15 | 0.56 | 0.29 |
| OLMo | (ii) | 0.689 | 0.35 | 0.34 | 0.31 |
| OLMo | (iii) | 0.672 | 0.44 | 0.22 | 0.34 |
| OLMo | reference | — | 0.38 | 0.44 | 0.18 |

| LLM | AURC (i) | AURC A | AURC (i)+A |
|---|---|---|---|
| Qwen | 0.212 | 0.261 | 0.212 |
| OLMo | 0.215 | 0.277 | 0.220 |

*Table S8.12. AURC over abstention rates 0–0.95 (lower is better) on all ranked items with a reference, with 95% unit-bootstrap intervals of the differences (flags fixed, as in the frozen risk–coverage analysis) [P].*

| Study | Classifier × LLM | AURC (i) | AURC A | AURC (i)+A | A − (i) | (i)+A − A |
|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | 0.191 | 0.144 | 0.145 | −0.047 [−0.069, −0.026] | 0.002 [−0.010, 0.013] |
| 1 | FLAN-T5 × Qwen | 0.434 | 0.237 | 0.292 | −0.197 [−0.226, −0.167] | 0.056 [0.040, 0.071] |
| 1 | DeBERTa × OLMo | 0.193 | 0.183 | 0.171 | −0.011 [−0.033, 0.012] | −0.011 [−0.026, 0.002] |
| 2 | DeBERTa × Qwen | 0.115 | 0.065 | 0.066 | −0.050 [−0.067, −0.034] | 0.001 [−0.006, 0.008] |
| 2 | FLAN-T5 × Qwen | 0.421 | 0.182 | 0.237 | −0.239 [−0.267, −0.211] | 0.054 [0.043, 0.066] |
| 2 | DeBERTa × OLMo | 0.115 | 0.121 | 0.093 | 0.006 [−0.014, 0.027] | −0.028 [−0.040, −0.016] |
| 2 | FLAN-T5 × OLMo | 0.421 | 0.273 | 0.302 | −0.148 [−0.178, −0.118] | 0.029 [0.014, 0.045] |
| 1 | ChaosNLI RoBERTa × Qwen | 0.212 | 0.261 | 0.212 | 0.049 [0.031, 0.067] | −0.049 [−0.060, −0.038] |
| 1 | ChaosNLI RoBERTa × OLMo | 0.215 | 0.277 | 0.220 | 0.062 [0.045, 0.080] | −0.057 [−0.067, −0.046] |

*Table S8.13. Confidence on stable errors against correctly classified items only (unstable errors left out) [P]: AUROC of (i) and A, (i) − 0.5, and (i)'s detection rate minus the random expectation 0.30 (unit bootstrap). (i) does not depend on the LLM; the Qwen rows are shown.*

| Study | Classifier | Correct items | AUROC (i) | AUROC A (Qwen) | (i) − 0.5 | Detection (i) − 0.30 |
|---|---|---|---|---|---|---|
| 1 | DeBERTa | 946 | 0.574 | 0.732 | 0.074 [0.040, 0.107] | 0.003 [−0.042, 0.050] |
| 1 | FLAN-T5 | 727 | 0.409 | 0.787 | −0.091 [−0.122, −0.058] | −0.113 [−0.148, −0.076] |
| 2 | DeBERTa | 1,306 | 0.626 | 0.831 | 0.126 [0.070, 0.181] | 0.080 [0.007, 0.158] |
| 2 | FLAN-T5 | 895 | 0.468 | 0.904 | −0.032 [−0.064, 0.000] | −0.094 [−0.130, −0.055] |

*Table S8.14. DeBERTa's stable errors on which both adjudicators chose the label opposite to the reference [P]: detection rates and A − (ii) (unit bootstrap). †: OLMo's (ii) is not a working judge. Study 2 has 8 such errors; its intervals are not interpretable.*

| Study | LLM | n (units) | A | (i) | (ii) | debiased (ii) | A − (ii) | A − debiased (ii) |
|---|---|---|---|---|---|---|---|---|
| 1 | Qwen | 68 (66) | 0.306 | 0.200 | 0.156 | 0.174 | 0.150 [0.049, 0.254] | 0.132 [0.029, 0.236] |
| 1 | OLMo † | 66 (64) | 0.315 | 0.203 | 0.188 | 0.185 | 0.127 [0.033, 0.224] | 0.130 [0.041, 0.221] |
| 2 | Qwen | 8 (8) | 0.325 | 0.375 | 0.575 | 0.575 | −0.250 [−0.600, 0.100] | −0.250 [−0.575, 0.100] |
| 2 | OLMo † | 8 (8) | 0.425 | 0.375 | 0.325 | 0.375 | 0.100 [−0.275, 0.475] | 0.050 [−0.225, 0.350] |

*Table S8.15. Stable-correct items of the adjudication sheet (DeBERTa) on which both raters chose the label opposite to the reference ("reversed"; not called errors, see S9) or the reference itself ("confirmed") [P]: flag rates and A − (ii) (unit bootstrap). †: OLMo's (ii) is not a working judge. Intervals are not given for a single item.*

| Study | LLM | Items | n (units) | A | (i) | (ii) | debiased (ii) | A − (ii) | A − debiased (ii) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen | reversed | 13 (13) | 0.662 | 0.462 | 0.585 | 0.492 | 0.077 [−0.154, 0.323] | 0.169 [−0.077, 0.431] |
| 1 | Qwen | confirmed | 180 (173) | 0.133 | 0.151 | 0.164 | 0.149 | −0.031 [−0.082, 0.020] | −0.016 [−0.065, 0.034] |
| 1 | OLMo † | reversed | 13 (13) | 0.554 | 0.462 | 0.508 | 0.338 | 0.046 [−0.169, 0.246] | 0.215 [−0.062, 0.462] |
| 1 | OLMo † | confirmed | 173 (167) | 0.206 | 0.143 | 0.203 | 0.214 | 0.002 [−0.058, 0.064] | −0.008 [−0.072, 0.056] |
| 2 | Qwen | reversed | 1 (1) | 1.000 | 0.000 | 0.400 | 0.200 | — | — |
| 2 | Qwen | confirmed | 89 (85) | 0.133 | 0.198 | 0.117 | 0.153 | 0.016 [−0.047, 0.076] | −0.020 [−0.089, 0.044] |
| 2 | OLMo † | reversed | 1 (1) | 0.200 | 0.000 | 0.000 | 0.000 | — | — |
| 2 | OLMo † | confirmed | 89 (85) | 0.247 | 0.198 | 0.088 | 0.216 | 0.160 [0.076, 0.243] | 0.031 [−0.061, 0.125] |

*Table S8.16. Study 1 contested stable errors of DeBERTa (not both raters chose the reference) by reference label and by whether both sentences are WordNet usage examples (P4) [P]: flag rates and A − (ii) (unit bootstrap). †: OLMo's (ii) is not a working judge.*

| LLM | Reference | Origin | n (units) | A | (ii) | debiased (ii) | A − (ii) | A − debiased (ii) |
|---|---|---|---|---|---|---|---|---|
| Qwen | same | WordNet examples | 11 (11) | 0.345 | 0.091 | 0.164 | 0.255 [0.055, 0.455] | 0.182 [−0.055, 0.418] |
| Qwen | same | other | 42 (41) | 0.305 | 0.071 | 0.224 | 0.233 [0.122, 0.348] | 0.081 [−0.052, 0.219] |
| Qwen | different | WordNet examples | 38 (37) | 0.400 | 0.389 | 0.316 | 0.011 [−0.130, 0.147] | 0.084 [−0.049, 0.216] |
| Qwen | different | other | 17 (17) | 0.329 | 0.318 | 0.224 | 0.012 [−0.176, 0.200] | 0.106 [−0.071, 0.282] |
| OLMo † | same | WordNet examples | 10 (10) | 0.540 | 0.000 | 0.420 | 0.540 [0.360, 0.720] | 0.120 [−0.200, 0.420] |
| OLMo † | same | other | 41 (40) | 0.220 | 0.000 | 0.137 | 0.220 [0.135, 0.310] | 0.083 [−0.040, 0.200] |
| OLMo † | different | WordNet examples | 38 (37) | 0.432 | 0.511 | 0.321 | −0.079 [−0.215, 0.059] | 0.111 [−0.042, 0.263] |
| OLMo † | different | other | 17 (17) | 0.306 | 0.424 | 0.224 | −0.118 [−0.282, 0.035] | 0.082 [−0.082, 0.247] |

*Table S8.17. Flags restricted to stably predicted items (same label on the original, swapped and paraphrased inputs; no labels used) [P]: share of ranked items that are stably predicted, detection rates (q-mean over q = 0.1–0.5) of (i) and A without and with the restriction (subscript r), and differences (unit bootstrap). (i) does not depend on the LLM. Baselines for the restriction (a random ranking restricted in the same way, the ceiling) are in Table S8.19.*

| Study | Classifier × LLM | Stable predictions | (i) | A | (i)_r | A_r | A_r − (i)_r | A − (i)_r | (i)_r − (i) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | 824 of 1,304 | 0.303 | 0.509 | 0.592 | 0.698 | 0.106 [0.049, 0.162] | −0.084 [−0.146, −0.023] | 0.289 [0.267, 0.312] |
| 1 | FLAN-T5 × Qwen | 714 of 1,304 | 0.187 | 0.463 | 0.534 | 0.741 | 0.206 [0.161, 0.251] | −0.072 [−0.125, −0.017] | 0.347 [0.324, 0.370] |
| 1 | DeBERTa × OLMo | 796 of 1,270 | 0.302 | 0.391 | 0.595 | 0.626 | 0.031 [−0.025, 0.087] | −0.204 [−0.264, −0.144] | 0.293 [0.270, 0.316] |
| 2 | DeBERTa × Qwen | 657 of 1,653 | 0.380 | 0.648 | 0.809 | 0.903 | 0.095 [0.039, 0.149] | −0.160 [−0.250, −0.074] | 0.429 [0.383, 0.472] |
| 2 | FLAN-T5 × Qwen | 630 of 1,653 | 0.206 | 0.556 | 0.732 | 0.874 | 0.142 [0.111, 0.176] | −0.175 [−0.227, −0.122] | 0.526 [0.502, 0.548] |
| 2 | DeBERTa × OLMo | 657 of 1,653 | 0.380 | 0.508 | 0.809 | 0.824 | 0.015 [−0.048, 0.076] | −0.301 [−0.391, −0.212] | 0.429 [0.383, 0.472] |
| 2 | FLAN-T5 × OLMo | 630 of 1,653 | 0.206 | 0.427 | 0.732 | 0.819 | 0.087 [0.052, 0.123] | −0.305 [−0.359, −0.247] | 0.526 [0.502, 0.548] |

*Table S8.18. A with the coarsening of WiC test's reference (A_c: first-degree WordNet neighbours, sister senses and senses of the same supersense count as the same sense; no labels used) [P]: share of ranked items predicted different, detection rates and differences (unit bootstrap). †: OLMo's (ii) is not a working judge.*

| Study | Classifier × LLM | Different: A | Different: A_c | A | A_c | A_c − A | A_c − (i) | A_c − (ii) | A_c − debiased (ii) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | 0.571 | 0.501 | 0.509 | 0.559 | 0.050 [0.017, 0.085] | 0.256 [0.194, 0.318] | 0.204 [0.140, 0.267] | 0.175 [0.114, 0.236] |
| 1 | FLAN-T5 × Qwen | 0.571 | 0.501 | 0.463 | 0.500 | 0.038 [0.014, 0.063] | 0.313 [0.258, 0.367] | 0.153 [0.103, 0.202] | 0.116 [0.070, 0.162] |
| 1 | DeBERTa × OLMo † | 0.624 | 0.494 | 0.391 | 0.438 | 0.047 [0.015, 0.081] | 0.136 [0.074, 0.198] | 0.121 [0.051, 0.192] | 0.102 [0.038, 0.167] |
| 2 | DeBERTa × Qwen | 0.503 | 0.356 | 0.648 | 0.492 | −0.156 [−0.230, −0.089] | 0.112 [0.000, 0.222] | −0.088 [−0.214, 0.045] | −0.090 [−0.210, 0.035] |
| 2 | FLAN-T5 × Qwen | 0.503 | 0.356 | 0.556 | 0.494 | −0.062 [−0.092, −0.031] | 0.288 [0.229, 0.346] | 0.014 [−0.047, 0.074] | −0.020 [−0.080, 0.039] |
| 2 | DeBERTa × OLMo † | 0.523 | 0.357 | 0.508 | 0.380 | −0.127 [−0.190, −0.071] | 0.000 [−0.104, 0.102] | −0.095 [−0.219, 0.034] | −0.053 [−0.170, 0.067] |
| 2 | FLAN-T5 × OLMo † | 0.523 | 0.357 | 0.427 | 0.414 | −0.013 [−0.049, 0.024] | 0.208 [0.146, 0.269] | 0.119 [0.056, 0.182] | −0.035 [−0.103, 0.034] |

*Table S8.19. Baselines for the flags restricted to stably predicted items (P22) [P]: a random ranking restricted in the same way (random_r), the ceiling (perfect ranking), detection rates of the restricted methods, normalised detection (detection − random_r)/(ceiling − random_r), and differences (unit bootstrap). q = 0.1–0.3: mean over the three flag rates at which the budget does not exceed the stably predicted items (all five in Study 1). (ii) columns for Qwen only; †: OLMo's (ii) is not a working judge.*

| Study | Classifier × LLM | random_r | Ceiling | (i)_r | A_r | (ii)_r | Debiased (ii)_r | Normalised (i)_r | Normalised A_r | (i)_r − random_r | A_r − random_r | A_r − debiased (ii)_r | A_r − (i)_r, q = 0.1–0.3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | 0.475 | 0.925 | 0.592 | 0.698 | 0.603 | 0.605 | 0.261 | 0.496 | 0.118 [0.073, 0.162] | 0.223 [0.179, 0.267] | 0.093 [0.040, 0.146] | 0.106 [0.049, 0.162] |
| 1 | FLAN-T5 × Qwen | 0.548 | 0.867 | 0.534 | 0.741 | 0.689 | 0.698 | −0.042 | 0.604 | −0.013 [−0.049, 0.023] | 0.193 [0.162, 0.222] | 0.043 [0.010, 0.076] | 0.206 [0.161, 0.251] |
| 1 | DeBERTa × OLMo † | 0.479 | 0.923 | 0.595 | 0.626 | — | — | 0.262 | 0.332 | 0.117 [0.074, 0.160] | 0.148 [0.104, 0.190] | — | 0.031 [−0.025, 0.087] |
| 2 | DeBERTa × Qwen | 0.702 | 1.000 | 0.809 | 0.903 | 0.857 | 0.859 | 0.358 | 0.676 | 0.107 [0.064, 0.150] | 0.201 [0.162, 0.237] | 0.044 [−0.009, 0.098] | 0.158 [0.065, 0.248] |
| 2 | FLAN-T5 × Qwen | 0.715 | 0.916 | 0.732 | 0.874 | 0.860 | 0.865 | 0.085 | 0.793 | 0.017 [−0.010, 0.043] | 0.159 [0.141, 0.178] | 0.010 [−0.015, 0.035] | 0.237 [0.185, 0.293] |
| 2 | DeBERTa × OLMo † | 0.702 | 1.000 | 0.809 | 0.824 | — | — | 0.358 | 0.410 | 0.107 [0.064, 0.150] | 0.122 [0.077, 0.166] | — | 0.026 [−0.080, 0.127] |
| 2 | FLAN-T5 × OLMo † | 0.715 | 0.916 | 0.732 | 0.819 | — | — | 0.085 | 0.517 | 0.017 [−0.010, 0.043] | 0.104 [0.084, 0.125] | — | 0.145 [0.087, 0.205] |

*Table S8.20. The restriction on all eligible errors (stable or not) and on the risk–coverage curve over all ranked items with a reference (AURC, abstention 0–0.95, lower is better) (P22) [P].*

| Study | Classifier × LLM | All errors: (i) | All errors: (i)_r | All errors: A | All errors: A_r | All errors: A_r − (i)_r | AURC: (i) | AURC: (i)_r | AURC: A | AURC: A_r | AURC: A_r − (i)_r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | 0.403 | 0.437 | 0.536 | 0.515 | 0.078 [0.038, 0.119] | 0.191 | 0.251 | 0.144 | 0.215 | −0.036 [−0.050, −0.022] |
| 1 | FLAN-T5 × Qwen | 0.338 | 0.341 | 0.488 | 0.473 | 0.132 [0.102, 0.162] | 0.434 | 0.462 | 0.237 | 0.344 | −0.119 [−0.139, −0.098] |
| 1 | DeBERTa × OLMo | 0.403 | 0.439 | 0.419 | 0.462 | 0.023 [−0.019, 0.065] | 0.193 | 0.254 | 0.183 | 0.247 | −0.006 [−0.021, 0.009] |
| 2 | DeBERTa × Qwen | 0.533 | 0.529 | 0.703 | 0.593 | 0.064 [0.026, 0.102] | 0.115 | 0.174 | 0.065 | 0.131 | −0.043 [−0.057, −0.030] |
| 2 | FLAN-T5 × Qwen | 0.303 | 0.602 | 0.556 | 0.710 | 0.108 [0.079, 0.137] | 0.421 | 0.433 | 0.182 | 0.273 | −0.160 [−0.182, −0.139] |
| 2 | DeBERTa × OLMo | 0.533 | 0.529 | 0.508 | 0.529 | 0.000 [−0.041, 0.040] | 0.115 | 0.174 | 0.121 | 0.179 | 0.005 [−0.012, 0.021] |
| 2 | FLAN-T5 × OLMo | 0.303 | 0.602 | 0.429 | 0.662 | 0.059 [0.030, 0.090] | 0.421 | 0.433 | 0.273 | 0.345 | −0.089 [−0.113, −0.066] |

*Table S8.21. Study 2 with stability measured on the paraphrases as formed, before the human check (P22) [P]. Detection rates are q-means over q = 0.1–0.5 (the budget does not exceed the restricted items here).*

| Classifier × LLM | Stably predicted | random_r | (i)_r | A_r | A_r − (i)_r |
|---|---|---|---|---|---|
| DeBERTa × Qwen | 1,216 of 1,653 | 0.408 | 0.593 | 0.765 | 0.171 [0.076, 0.266] |
| FLAN-T5 × Qwen | 1,162 of 1,653 | 0.427 | 0.440 | 0.707 | 0.267 [0.212, 0.325] |
| DeBERTa × OLMo | 1,216 of 1,653 | 0.408 | 0.593 | 0.644 | 0.051 [−0.057, 0.153] |
| FLAN-T5 × OLMo | 1,162 of 1,653 | 0.427 | 0.440 | 0.592 | 0.152 [0.088, 0.215] |

*Table S8.22. Coarsened A with its share of "different" restored (P23), and coarse WiC (P24) [P]: share predicted different by A and A_c, detection rates, and differences (unit bootstrap). A_c back: A_c shifted to A's share by one log-odds constant. Debiased (ii) at A's share (as P6) or at A_c's share. Coarse WiC has 20 lemmas (20 units). †: OLMo's (ii) is not a working judge.*

| Study | Data | Classifier × LLM | Share different: A | Share different: A_c | Detection: A | Detection: A_c | Detection: A_c back | A_c − A | A_c back − A | A − debiased (ii) at A | A_c − debiased (ii) at A_c |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | WiC test | DeBERTa × Qwen | 0.571 | 0.501 | 0.509 | 0.559 | 0.551 | 0.050 [0.017, 0.085] | 0.042 [0.010, 0.075] | 0.125 [0.066, 0.185] | 0.175 [0.114, 0.236] |
| 1 | WiC test | FLAN-T5 × Qwen | 0.571 | 0.501 | 0.463 | 0.500 | 0.484 | 0.038 [0.014, 0.063] | 0.021 [−0.004, 0.047] | 0.078 [0.036, 0.121] | 0.098 [0.052, 0.145] |
| 1 | WiC test | DeBERTa × OLMo † | 0.624 | 0.494 | 0.391 | 0.438 | 0.437 | 0.047 [0.015, 0.081] | 0.046 [0.016, 0.075] | 0.055 [−0.005, 0.117] | 0.098 [0.037, 0.160] |
| 2 | MCL-WiC dev+test | DeBERTa × Qwen | 0.503 | 0.356 | 0.648 | 0.492 | 0.514 | −0.156 [−0.230, −0.089] | −0.134 [−0.214, −0.058] | 0.066 [−0.030, 0.167] | 0.022 [−0.088, 0.134] |
| 2 | MCL-WiC dev+test | FLAN-T5 × Qwen | 0.503 | 0.356 | 0.556 | 0.494 | 0.455 | −0.062 [−0.092, −0.031] | −0.102 [−0.136, −0.066] | 0.042 [−0.017, 0.100] | −0.047 [−0.103, 0.009] |
| 2 | MCL-WiC dev+test | DeBERTa × OLMo † | 0.523 | 0.357 | 0.508 | 0.380 | 0.411 | −0.127 [−0.190, −0.071] | −0.097 [−0.161, −0.037] | 0.075 [−0.038, 0.193] | 0.020 [−0.083, 0.123] |
| 2 | MCL-WiC dev+test | FLAN-T5 × OLMo † | 0.523 | 0.357 | 0.427 | 0.414 | 0.380 | −0.013 [−0.049, 0.024] | −0.047 [−0.084, −0.007] | −0.022 [−0.091, 0.048] | −0.067 [−0.133, 0.001] |
| 1 | Coarse WiC | DeBERTa × Qwen | 0.617 | 0.590 | 0.555 | 0.553 | 0.544 | −0.002 [−0.015, 0.008] | −0.011 [−0.037, −0.001] | 0.051 [−0.019, 0.101] | 0.042 [−0.045, 0.098] |

*Table S8.23. Adjudicated stable errors against the random expectation (P25) [P] (Study 1, DeBERTa): detection rate of each method, and A and (ii) minus 0.30 (the q-mean of a random ranking), unit bootstrap. Both opposite: both raters chose the label opposite to the reference (the classifier's); confirmed: both chose the reference. †: OLMo's (ii) is not a working judge.*

| LLM | Errors | n (units) | A | (i) | (ii) | Debiased (ii) | A − 0.30 | (ii) − 0.30 | A − (ii) | A − debiased (ii) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen | Both opposite | 68 (66) | 0.306 | 0.200 | 0.156 | 0.174 | 0.006 [−0.073, 0.086] | −0.144 [−0.204, −0.079] | 0.150 [0.049, 0.254] | 0.132 [0.029, 0.236] |
| Qwen | Confirmed | 100 (100) | 0.684 | 0.372 | 0.496 | 0.528 | 0.384 [0.316, 0.448] | 0.196 [0.124, 0.270] | 0.188 [0.104, 0.274] | 0.156 [0.070, 0.244] |
| OLMo † | Both opposite | 66 (64) | 0.315 | 0.203 | 0.188 | 0.185 | 0.015 [−0.063, 0.094] | −0.112 [−0.180, −0.038] | 0.127 [0.033, 0.224] | 0.130 [0.041, 0.221] |
| OLMo † | Confirmed | 100 (100) | 0.446 | 0.370 | 0.386 | 0.434 | 0.146 [0.072, 0.220] | 0.086 [0.010, 0.164] | 0.060 [−0.024, 0.146] | 0.012 [−0.074, 0.102] |

*Table S8.24. The restricted comparisons at q = 0.1–0.3 in every configuration (P26) [P]: in Study 2 the budget exceeds the stably predicted items at q = 0.4 and 0.5, where every restricted difference is 0, so q-means over q = 0.1–0.5 shrink Study 2's differences to 3/5. Share: ((ii)_r − (i)_r)/(A_r − (i)_r) as in P2 (95% interval; Qwen only). †: OLMo's (ii) is not a working judge.*

| Study | Classifier × LLM | random_r | (i)_r | A_r | A_r − random_r | (i)_r − random_r | A_r − (i)_r | A_r − (ii)_r | A_r − debiased (ii)_r | Share of debiased (ii)_r |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | DeBERTa × Qwen | 0.316 | 0.425 | 0.572 | 0.256 [0.201, 0.310] | 0.108 [0.055, 0.163] | 0.147 [0.076, 0.217] | 0.135 [0.071, 0.199] | 0.109 [0.043, 0.175] | 0.26 [−0.29, 0.64] |
| 1 | FLAN-T5 × Qwen | 0.365 | 0.346 | 0.603 | 0.238 [0.196, 0.279] | −0.019 [−0.063, 0.026] | 0.257 [0.198, 0.315] | 0.072 [0.022, 0.119] | 0.057 [0.011, 0.102] | 0.78 [0.61, 0.95] |
| 1 | DeBERTa × OLMo † | 0.319 | 0.427 | 0.463 | 0.144 [0.088, 0.197] | 0.108 [0.055, 0.162] | 0.036 [−0.036, 0.105] | — | — | — |
| 2 | DeBERTa × Qwen | 0.503 | 0.681 | 0.839 | 0.336 [0.271, 0.396] | 0.178 [0.106, 0.250] | 0.158 [0.065, 0.248] | 0.077 [−0.011, 0.170] | 0.073 [−0.014, 0.163] | 0.53 [0.04, 1.14] |
| 2 | FLAN-T5 × Qwen | 0.525 | 0.553 | 0.791 | 0.266 [0.235, 0.296] | 0.028 [−0.016, 0.072] | 0.237 [0.185, 0.293] | 0.025 [−0.019, 0.067] | 0.016 [−0.026, 0.058] | 0.93 [0.77, 1.12] |
| 2 | DeBERTa × OLMo † | 0.503 | 0.681 | 0.707 | 0.204 [0.129, 0.277] | 0.178 [0.106, 0.250] | 0.026 [−0.080, 0.127] | — | — | — |
| 2 | FLAN-T5 × OLMo † | 0.525 | 0.553 | 0.698 | 0.173 [0.139, 0.208] | 0.028 [−0.016, 0.072] | 0.145 [0.087, 0.205] | — | — | — |
