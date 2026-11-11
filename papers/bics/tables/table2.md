*Table 5. From input items to stable errors (top), and the factors of the preregistered expectation for Study 2 (bottom). Excluded items stay in the ranking of every method but not in the error sets. Stop rule: at least 100 stable errors and 50 units for the primary classifier. All MCL-WiC nouns and verbs have at least two WordNet 3.0 senses, hence the 0.*

| Study | Data | Classifier | Items | < 2 WordNet senses | Paraphrase not formed | Rejected by check | Reference tie | Errors (stable + unstable) | Stable errors (units) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | WiC test | DeBERTa | 1,400 | 96 | 261 | 0 | 0 | 282 | 208 (201) |
| 1 | WiC test | FLAN-T5 | 1,400 | 96 | 261 | 0 | 0 | 459 | 293 (280) |
| 1 | ChaosNLI | RoBERTa | 3,113 | 0 | 3 | 0 | 28 | 973 | 699 (671) |
| 2 | MCL-WiC dev+test | DeBERTa | 1,654 | 0 | 197 | 684 | 0 | 150 | 91 (81) |
| 2 | MCL-WiC dev+test | FLAN-T5 | 1,654 | 0 | 197 | 684 | 0 | 360 | 285 (225) |

| Study 2 factor | DeBERTa: projected → observed (ratio) | FLAN-T5: projected → observed (ratio) |
|---|---|---|
| base error rate | 0.257 → 0.210 (0.82) | 0.457 → 0.458 (1.00) |
| paraphrase 1 formed | 0.865 → 0.881 (1.02) | 0.865 → 0.881 (1.02) |
| passed human check | 0.575 → 0.531 (0.92) | 0.575 → 0.531 (0.92) |
| errors vs. check exclusion (correlation term) | 1.000 → 0.922 (0.92) | 1.000 → 1.016 (1.02) |
| stable share of errors | 0.740 → 0.607 (0.82) | 0.640 → 0.792 (1.24) |
| stable errors (product x 1,654) | 156.5 → 91 (0.58) | 240.6 → 285 (1.18) |
