# Eval Table Comparison

Generated from `Experiments/eval/*/{batch,pair,score}/{summary.json,bias_summary.json}`.

## Metric Tables

### Batch

| family | model | size_label | accuracy | pairwise_agreement_mean | kendall_tau_like_mean | sequence_confidence_mean | auroc_uncertainty_for_incorrect | ece | selection_avg_kept_accuracy_alpha_0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma | google_gemma-3-4b-it | 4B | 0.1194 | 0.5715 | 0.1430 | 0.3007 | 0.6135 | 0.1812 | 0.1194 |
| Gemma | google_gemma-3-12b-it | 12B | 0.0893 | 0.6636 | 0.3272 | 0.3418 | 0.5681 | 0.2524 | 0.0893 |
| Gemma | google_gemma-3-27b-it | 27B | 0.1330 | 0.6660 | 0.3320 | 0.3985 | 0.4931 | 0.2655 | 0.1330 |
| Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 0.1223 | 0.5390 | 0.1275 | 0.3467 | 0.6957 | 0.2244 | 0.1287 |
| Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 0.2233 | 0.6531 | 0.3557 | 0.6798 | 0.6425 | 0.4572 | 0.2349 |
| Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 0.1660 | 0.6173 | 0.2841 | 0.4610 | 0.6405 | 0.2949 | 0.1770 |

### Pair

| family | model | size_label | accuracy | nll | brier | entropy_mean | auroc_uncertainty_for_incorrect | ece | conformal_coverage_alpha_0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma | google_gemma-3-4b-it | 4B | 0.4774 | 4.5134 | 1.0194 | 0.0552 | 0.5507 | 0.4850 | 0.9162 |
| Gemma | google_gemma-3-12b-it | 12B | 0.5513 | 3.8837 | 0.8653 | 0.0551 | 0.5648 | 0.5481 | 0.9118 |
| Gemma | google_gemma-3-27b-it | 27B | 0.5502 | 4.9099 | 0.8762 | 0.0396 | 0.5957 | 0.5453 | 0.9129 |
| Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 0.4983 | 2.5970 | 0.8566 | 0.2492 | 0.6395 | 0.4998 | 0.9239 |
| Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 0.4598 | 3.1067 | 0.8943 | 0.2889 | 0.6769 | 0.4905 | 0.9195 |
| Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 0.5127 | 1.8371 | 0.7175 | 0.4625 | 0.6538 | 0.3464 | 0.9228 |

### Score

| family | model | size_label | accuracy | nll | brier | entropy_mean | auroc_uncertainty_for_incorrect | ece | conformal_coverage_alpha_0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma | google_gemma-3-4b-it | 4B | 0.3476 | 6.8920 | 1.2430 | 0.1239 | 0.4928 | 0.6001 | 0.8934 |
| Gemma | google_gemma-3-12b-it | 12B | 0.3287 | 8.0321 | 1.2636 | 0.1224 | 0.5673 | 0.6227 | 0.8984 |
| Gemma | google_gemma-3-27b-it | 27B | 0.3227 | 10.8352 | 1.3130 | 0.0693 | 0.5773 | 0.6484 | 0.9990 |
| Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 0.2938 | 5.0938 | 1.1415 | 0.4396 | 0.5374 | 0.5066 | 0.9183 |
| Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 0.3028 | 5.9638 | 1.1457 | 0.3945 | 0.5951 | 0.5237 | 0.9124 |
| Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 0.2709 | 3.0083 | 1.0375 | 0.6989 | 0.6074 | 0.4131 | 0.9124 |

## Bias Tables

### by_original_dataset

| task | family | model | size_label | n | accuracy | avg_confidence |
| --- | --- | --- | --- | --- | --- | --- |
| batch | Gemma | google_gemma-3-4b-it | 4B | 1030 | 0.1263 | 0.3029 |
| pair | Gemma | google_gemma-3-4b-it | 4B | 907 | 0.4990 | 0.0086 |
| score | Gemma | google_gemma-3-4b-it | 4B | 1004 | 0.3528 | 0.9488 |
| batch | Gemma | google_gemma-3-12b-it | 12B | 1030 | 0.0996 | 0.3456 |
| pair | Gemma | google_gemma-3-12b-it | 12B | 907 | 0.5578 | 0.0686 |
| score | Gemma | google_gemma-3-12b-it | 12B | 1004 | 0.3396 | 0.9530 |
| batch | Gemma | google_gemma-3-27b-it | 27B | 1030 | 0.1387 | 0.3967 |
| pair | Gemma | google_gemma-3-27b-it | 27B | 907 | 0.5551 | 0.0664 |
| score | Gemma | google_gemma-3-27b-it | 27B | 1004 | 0.3280 | 0.9725 |
| batch | Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 979 | 0.1314 | 0.3556 |
| pair | Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 869 | 0.5251 | 0.0161 |
| score | Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 967 | 0.3097 | 0.8347 |
| batch | Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 979 | 0.2342 | 0.7027 |
| pair | Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 869 | 0.4619 | 0.2199 |
| score | Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 967 | 0.3156 | 0.8658 |
| batch | Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 979 | 0.1751 | 0.4833 |
| pair | Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 869 | 0.5420 | 0.6269 |
| score | Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 967 | 0.2777 | 0.7167 |

### by_predicted_top_name

| task | family | model | size_label | n | accuracy | avg_confidence |
| --- | --- | --- | --- | --- | --- | --- |
| batch | Gemma | google_gemma-3-4b-it | 4B | 1030 | 0.1837 | 0.3494 |
| pair | Gemma | google_gemma-3-4b-it | 4B | 907 | 0.4741 | 0.0512 |
| score | Gemma | google_gemma-3-4b-it | 4B | 1004 | 0.3498 | 0.9429 |
| batch | Gemma | google_gemma-3-12b-it | 12B | 1030 | 0.0754 | 0.3617 |
| pair | Gemma | google_gemma-3-12b-it | 12B | 907 | 0.4527 | 0.1241 |
| score | Gemma | google_gemma-3-12b-it | 12B | 1004 | 0.3458 | 0.9509 |
| batch | Gemma | google_gemma-3-27b-it | 27B | 1030 | 0.0811 | 0.4602 |
| pair | Gemma | google_gemma-3-27b-it | 27B | 907 | 0.4849 | 0.0946 |
| score | Gemma | google_gemma-3-27b-it | 27B | 1004 | 0.3548 | 0.9759 |
| batch | Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 979 | 0.1354 | 0.4076 |
| pair | Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 869 | 0.4272 | 0.0610 |
| score | Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 967 | 0.3142 | 0.8266 |
| batch | Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 979 | 0.1168 | 0.6455 |
| pair | Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 869 | 0.4164 | 0.2435 |
| score | Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 967 | 0.3252 | 0.8590 |
| batch | Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 979 | 0.1135 | 0.4666 |
| pair | Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 869 | 0.4650 | 0.6335 |
| score | Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 967 | 0.3037 | 0.7153 |

### by_human_top_name

| task | family | model | size_label | n | accuracy | avg_confidence |
| --- | --- | --- | --- | --- | --- | --- |
| batch | Gemma | google_gemma-3-4b-it | 4B | 1030 | 0.1012 | 0.3065 |
| pair | Gemma | google_gemma-3-4b-it | 4B | 907 | 0.4748 | 0.0141 |
| score | Gemma | google_gemma-3-4b-it | 4B | 1004 | 0.3498 | 0.9429 |
| batch | Gemma | google_gemma-3-12b-it | 12B | 1030 | 0.0680 | 0.3659 |
| pair | Gemma | google_gemma-3-12b-it | 12B | 907 | 0.4970 | 0.0608 |
| score | Gemma | google_gemma-3-12b-it | 12B | 1004 | 0.3458 | 0.9509 |
| batch | Gemma | google_gemma-3-27b-it | 27B | 1030 | 0.0803 | 0.3984 |
| pair | Gemma | google_gemma-3-27b-it | 27B | 907 | 0.5031 | 0.0934 |
| score | Gemma | google_gemma-3-27b-it | 27B | 1004 | 0.3548 | 0.9759 |
| batch | Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 979 | 0.0624 | 0.3314 |
| pair | Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 869 | 0.4862 | 0.0217 |
| score | Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 967 | 0.3142 | 0.8266 |
| batch | Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 979 | 0.0965 | 0.6800 |
| pair | Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 869 | 0.4495 | 0.2458 |
| score | Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 967 | 0.3252 | 0.8590 |
| batch | Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 979 | 0.1066 | 0.5054 |
| pair | Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 869 | 0.5111 | 0.6333 |
| score | Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 967 | 0.3037 | 0.7153 |

### by_name

| task | family | model | size_label | n | accuracy | avg_confidence |
| --- | --- | --- | --- | --- | --- | --- |
| batch | Gemma | google_gemma-3-4b-it | 4B | 1030 | 0.1194 | 0.3007 |
| pair | Gemma | google_gemma-3-4b-it | 4B | 907 | 0.4774 | 0.0134 |
| score | Gemma | google_gemma-3-4b-it | 4B | 1004 | 0.3498 | 0.9429 |
| batch | Gemma | google_gemma-3-12b-it | 12B | 1030 | 0.0893 | 0.3418 |
| pair | Gemma | google_gemma-3-12b-it | 12B | 907 | 0.5513 | 0.0674 |
| score | Gemma | google_gemma-3-12b-it | 12B | 1004 | 0.3458 | 0.9509 |
| batch | Gemma | google_gemma-3-27b-it | 27B | 1030 | 0.1330 | 0.3985 |
| pair | Gemma | google_gemma-3-27b-it | 27B | 907 | 0.5502 | 0.0857 |
| score | Gemma | google_gemma-3-27b-it | 27B | 1004 | 0.3548 | 0.9759 |
| batch | Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 979 | 0.1287 | 0.3648 |
| pair | Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 869 | 0.5201 | 0.0161 |
| score | Qwen | Qwen_Qwen3-VL-4B-Instruct | 4B | 967 | 0.3142 | 0.8266 |
| batch | Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 979 | 0.2349 | 0.7153 |
| pair | Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 869 | 0.4799 | 0.2319 |
| score | Qwen | Qwen_Qwen3-VL-8B-Instruct | 8B | 967 | 0.3252 | 0.8590 |
| batch | Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 979 | 0.1747 | 0.4850 |
| pair | Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 869 | 0.5351 | 0.6292 |
| score | Qwen | Qwen_Qwen3-VL-30B-A3B-Instruct | 30B | 967 | 0.3037 | 0.7153 |
