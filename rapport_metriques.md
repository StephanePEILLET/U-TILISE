# Rapport de Métriques — Cloud Reconstruction U-TILISE

**Source des résultats :** logs SLURM dans `/mnt/stores/store_dai/tmp/speillet/logs` + JSON dans `/mnt/DATA_10T/data_rpg/outputs/U-TILISE/metrics`

## 1. Récapitulatif Global

### Masquage : Random Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 114.7 | 207.7 | 35.27 | 0.7129 | 0.0385 | 0.9535 |
| asc+desc, random_fully_masked, DA | 126.9 | 250.0 | 34.54 | 0.7202 | 0.0404 | 0.9358 |
| coherence_only | 268.9 | 509.7 | 30.91 | 0.7048 | 0.0696 | 0.7959 |
| **v2** mix_closest, random_fully_masked | 129.7 | 248.6 | 34.30 | 0.7234 | 0.0417 | 0.9364 |
| **v2** mix_closest, random_clouds | 107.4 | 204.7 | 35.69 | 0.7326 | 0.0353 | 0.9527 |
| **v2** asc+desc, random_fully_masked | 129.4 | 250.8 | 34.33 | 0.7200 | 0.0408 | 0.9356 |
| **v2** asc+desc, random_clouds | 116.0 | 216.0 | 35.06 | 0.7091 | 0.0392 | 0.9478 |
| **v3** mix_closest, random_clouds | 122.6 | 226.6 | 34.66 | 0.7039 | 0.0420 | 0.9432 |
| **v3** loss (L1+SSIM+L1_occ) | 110.8 | 206.7 | 35.46 | 0.7482 | 0.0368 | 0.9520 |
| **v3** wider | 113.3 | 215.5 | 35.22 | 0.7194 | 0.0366 | 0.9480 |
| **v3** combined (wider+cyclic+loss) | 114.1 | 211.1 | 35.14 | 0.7377 | 0.0374 | 0.9502 |
| **v3** cyclic | 108.0 | 210.1 | 35.62 | 0.7319 | 0.0355 | 0.9511 |
| **v4** asc+desc, combined | 110.0 | 194.3 | 35.55 | 0.7378 | 0.0353 | 0.9566 |
| **v4** mix_closest, combined | 119.9 | 220.5 | 34.76 | 0.7296 | 0.0393 | 0.9476 |

### Masquage : Consecutive Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 131.9 | 278.3 | 33.42 | 0.6548 | 0.0411 | 0.9187 |
| asc+desc, random_fully_masked, DA | 145.6 | 321.6 | 33.47 | 0.6832 | 0.0395 | 0.9077 |
| coherence_only | 389.1 | 837.2 | 26.54 | 0.6488 | 0.1002 | 0.6212 |
| **v2** mix_closest, random_fully_masked | 151.2 | 329.0 | 33.10 | 0.6783 | 0.0412 | 0.9038 |
| **v2** mix_closest, random_clouds | 117.0 | 262.6 | 34.26 | 0.6824 | 0.0360 | 0.9236 |
| **v2** asc+desc, random_fully_masked | 152.8 | 336.6 | 33.02 | 0.6760 | 0.0403 | 0.9000 |
| **v2** asc+desc, random_clouds | 125.6 | 270.7 | 33.75 | 0.6586 | 0.0393 | 0.9204 |
| **v3** mix_closest, random_clouds | 135.8 | 286.3 | 33.27 | 0.6651 | 0.0419 | 0.9132 |
| **v3** loss (L1+SSIM+L1_occ) | 125.2 | 269.3 | 33.87 | 0.6987 | 0.0377 | 0.9219 |
| **v3** wider | 130.6 | 287.7 | 33.45 | 0.6702 | 0.0376 | 0.9139 |
| **v3** combined (wider+cyclic+loss) | 122.9 | 266.0 | 33.82 | 0.6901 | 0.0377 | 0.9241 |
| **v3** cyclic | 119.1 | 273.0 | 34.09 | 0.6826 | 0.0354 | 0.9201 |
| **v4** asc+desc, combined | — | — | — | — | — | — |
| **v4** mix_closest, combined | — | — | — | — | — | — |

## 2. Métriques sur pixels reconstruits (Occluded)

### Masquage : Random Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 352.7 | 520.1 | 26.70 | 0.4409 | 0.0886 | 0.7902 |
| asc+desc, random_fully_masked, DA | 435.8 | 658.9 | 25.26 | 0.4019 | 0.1014 | 0.7359 |
| coherence_only | 1078.2 | 1382.8 | 19.53 | 0.3466 | 0.2360 | 0.4474 |
| **v2** mix_closest, random_fully_masked | 421.7 | 636.5 | 25.48 | 0.4587 | 0.0976 | 0.7420 |
| **v2** mix_closest, random_clouds | 360.5 | 538.3 | 26.36 | 0.4566 | 0.0903 | 0.7770 |
| **v2** asc+desc, random_fully_masked | 432.5 | 651.6 | 25.34 | 0.4591 | 0.0980 | 0.7394 |
| **v2** asc+desc, random_clouds | 379.2 | 556.4 | 26.01 | 0.4229 | 0.0965 | 0.7644 |
| **v3** mix_closest, random_clouds | 406.1 | 585.0 | 25.54 | 0.3765 | 0.1065 | 0.7422 |
| **v3** loss (L1+SSIM+L1_occ) | 351.0 | 528.1 | 26.60 | 0.4860 | 0.0861 | 0.7850 |
| **v3** wider | 374.1 | 560.7 | 26.10 | 0.4633 | 0.0891 | 0.7654 |
| **v3** combined (wider+cyclic+loss) | 353.2 | 532.8 | 26.48 | 0.4803 | 0.0869 | 0.7823 |
| **v3** cyclic | 381.0 | 563.5 | 25.98 | 0.4270 | 0.0981 | 0.7623 |
| **v4** asc+desc, combined | 277.0 | 439.9 | 28.52 | 0.5383 | 0.0636 | 0.8378 |
| **v4** mix_closest, combined | 376.9 | 559.3 | 26.04 | 0.4705 | 0.0934 | 0.7686 |

### Masquage : Consecutive Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 594.2 | 880.5 | 22.20 | 0.2190 | 0.1284 | 0.5547 |
| asc+desc, random_fully_masked, DA | 698.2 | 1026.0 | 22.28 | 0.2904 | 0.1122 | 0.5836 |
| coherence_only | 2665.8 | 2955.5 | 10.97 | 0.1041 | 0.5966 | 0.1401 |
| **v2** mix_closest, random_fully_masked | 704.1 | 1036.2 | 22.24 | 0.3089 | 0.1106 | 0.5790 |
| **v2** mix_closest, random_clouds | 542.1 | 847.0 | 22.71 | 0.2766 | 0.1151 | 0.5809 |
| **v2** asc+desc, random_fully_masked | 729.6 | 1070.8 | 21.91 | 0.3149 | 0.1111 | 0.5681 |
| **v2** asc+desc, random_clouds | 567.2 | 858.5 | 22.49 | 0.2374 | 0.1191 | 0.5722 |
| **v3** mix_closest, random_clouds | 626.4 | 907.3 | 21.91 | 0.2504 | 0.1291 | 0.5601 |
| **v3** loss (L1+SSIM+L1_occ) | 568.2 | 855.1 | 22.60 | 0.3146 | 0.1093 | 0.5880 |
| **v3** wider | 619.6 | 926.9 | 21.83 | 0.2896 | 0.1141 | 0.5720 |
| **v3** combined (wider+cyclic+loss) | 521.7 | 829.7 | 22.99 | 0.3176 | 0.1063 | 0.5993 |
| **v3** cyclic | 583.8 | 894.3 | 22.18 | 0.2528 | 0.1187 | 0.5675 |
| **v4** asc+desc, combined | — | — | — | — | — | — |
| **v4** mix_closest, combined | — | — | — | — | — | — |

## 3. Occluded vs Observed (détail)

### Masquage : Random Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | 352.7 | 520.1 | 26.70 | 0.4409 | 0.0886 | 0.7902 |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | 69.2 | 99.0 | 40.19 | 0.8106 | 0.0288 | 0.9911 |
| asc+desc, random_fully_masked, DA | Occluded | 435.8 | 658.9 | 25.26 | 0.4019 | 0.1014 | 0.7359 |
| asc+desc, random_fully_masked, DA | Observed | 67.7 | 98.3 | 40.33 | 0.8276 | 0.0290 | 0.9891 |
| coherence_only | Occluded | 1078.2 | 1382.8 | 19.53 | 0.3466 | 0.2360 | 0.4474 |
| coherence_only | Observed | 71.3 | 103.8 | 39.83 | 0.8044 | 0.0299 | 0.9883 |
| **v2** mix_closest, random_fully_masked | Occluded | 421.7 | 636.5 | 25.48 | 0.4587 | 0.0976 | 0.7420 |
| **v2** mix_closest, random_fully_masked | Observed | 73.3 | 106.3 | 39.63 | 0.8173 | 0.0311 | 0.9867 |
| **v2** mix_closest, random_clouds | Occluded | 360.5 | 538.3 | 26.36 | 0.4566 | 0.0903 | 0.7770 |
| **v2** mix_closest, random_clouds | Observed | 59.7 | 85.5 | 41.51 | 0.8306 | 0.0249 | 0.9922 |
| **v2** asc+desc, random_fully_masked | Occluded | 432.5 | 651.6 | 25.34 | 0.4591 | 0.0980 | 0.7394 |
| **v2** asc+desc, random_fully_masked | Observed | 71.5 | 103.5 | 39.87 | 0.8123 | 0.0301 | 0.9875 |
| **v2** asc+desc, random_clouds | Occluded | 379.2 | 556.4 | 26.01 | 0.4229 | 0.0965 | 0.7644 |
| **v2** asc+desc, random_clouds | Observed | 66.2 | 96.1 | 40.52 | 0.8095 | 0.0282 | 0.9894 |
| **v3** mix_closest, random_clouds | Occluded | 406.1 | 585.0 | 25.54 | 0.3765 | 0.1065 | 0.7422 |
| **v3** mix_closest, random_clouds | Observed | 68.9 | 100.0 | 40.19 | 0.8142 | 0.0298 | 0.9883 |
| **v3** loss (L1+SSIM+L1_occ) | Occluded | 351.0 | 528.1 | 26.60 | 0.4860 | 0.0861 | 0.7850 |
| **v3** loss (L1+SSIM+L1_occ) | Observed | 65.3 | 93.5 | 40.75 | 0.8428 | 0.0274 | 0.9898 |
| **v3** wider | Occluded | 374.1 | 560.7 | 26.10 | 0.4633 | 0.0891 | 0.7654 |
| **v3** wider | Observed | 64.2 | 92.9 | 40.79 | 0.8114 | 0.0267 | 0.9905 |
| **v3** combined (wider+cyclic+loss) | Occluded | 353.2 | 532.8 | 26.48 | 0.4803 | 0.0869 | 0.7823 |
| **v3** combined (wider+cyclic+loss) | Observed | 69.1 | 99.3 | 40.21 | 0.8307 | 0.0280 | 0.9887 |
| **v3** cyclic | Occluded | 381.0 | 563.5 | 25.98 | 0.4270 | 0.0981 | 0.7623 |
| **v3** cyclic | Observed | 56.7 | 82.4 | 41.80 | 0.8374 | 0.0237 | 0.9932 |
| **v4** asc+desc, combined | Occluded | 277.0 | 439.9 | 28.52 | 0.5383 | 0.0636 | 0.8378 |
| **v4** asc+desc, combined | Observed | 78.1 | 111.6 | 39.23 | 0.8173 | 0.0297 | 0.9858 |
| **v4** mix_closest, combined | Occluded | 376.9 | 559.3 | 26.04 | 0.4705 | 0.0934 | 0.7686 |
| **v4** mix_closest, combined | Observed | 71.7 | 102.5 | 39.94 | 0.8227 | 0.0291 | 0.9879 |

### Masquage : Consecutive Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | 594.2 | 880.5 | 22.20 | 0.2190 | 0.1284 | 0.5547 |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | 69.4 | 99.4 | 40.16 | 0.8098 | 0.0289 | 0.9910 |
| asc+desc, random_fully_masked, DA | Occluded | 698.2 | 1026.0 | 22.28 | 0.2904 | 0.1122 | 0.5836 |
| asc+desc, random_fully_masked, DA | Observed | 67.9 | 98.5 | 40.31 | 0.8272 | 0.0290 | 0.9890 |
| coherence_only | Occluded | 2665.8 | 2955.5 | 10.97 | 0.1041 | 0.5966 | 0.1401 |
| coherence_only | Observed | 71.6 | 104.2 | 39.79 | 0.8038 | 0.0300 | 0.9883 |
| **v2** mix_closest, random_fully_masked | Occluded | 704.1 | 1036.2 | 22.24 | 0.3089 | 0.1106 | 0.5790 |
| **v2** mix_closest, random_fully_masked | Observed | 73.7 | 106.7 | 39.60 | 0.8166 | 0.0311 | 0.9867 |
| **v2** mix_closest, random_clouds | Occluded | 542.1 | 847.0 | 22.71 | 0.2766 | 0.1151 | 0.5809 |
| **v2** mix_closest, random_clouds | Observed | 59.9 | 85.8 | 41.47 | 0.8300 | 0.0249 | 0.9922 |
| **v2** asc+desc, random_fully_masked | Occluded | 729.6 | 1070.8 | 21.91 | 0.3149 | 0.1111 | 0.5681 |
| **v2** asc+desc, random_fully_masked | Observed | 71.9 | 104.1 | 39.82 | 0.8117 | 0.0302 | 0.9875 |
| **v2** asc+desc, random_clouds | Occluded | 567.2 | 858.5 | 22.49 | 0.2374 | 0.1191 | 0.5722 |
| **v2** asc+desc, random_clouds | Observed | 66.4 | 96.4 | 40.48 | 0.8088 | 0.0283 | 0.9893 |
| **v3** mix_closest, random_clouds | Occluded | 626.4 | 907.3 | 21.91 | 0.2504 | 0.1291 | 0.5601 |
| **v3** mix_closest, random_clouds | Observed | 69.2 | 100.4 | 40.15 | 0.8135 | 0.0298 | 0.9883 |
| **v3** loss (L1+SSIM+L1_occ) | Occluded | 568.2 | 855.1 | 22.60 | 0.3146 | 0.1093 | 0.5880 |
| **v3** loss (L1+SSIM+L1_occ) | Observed | 65.6 | 94.0 | 40.70 | 0.8422 | 0.0275 | 0.9897 |
| **v3** wider | Occluded | 619.6 | 926.9 | 21.83 | 0.2896 | 0.1141 | 0.5720 |
| **v3** wider | Observed | 64.6 | 93.4 | 40.74 | 0.8106 | 0.0268 | 0.9904 |
| **v3** combined (wider+cyclic+loss) | Occluded | 521.7 | 829.7 | 22.99 | 0.3176 | 0.1063 | 0.5993 |
| **v3** combined (wider+cyclic+loss) | Observed | 69.5 | 99.8 | 40.17 | 0.8299 | 0.0280 | 0.9887 |
| **v3** cyclic | Occluded | 583.8 | 894.3 | 22.18 | 0.2528 | 0.1187 | 0.5675 |
| **v3** cyclic | Observed | 57.0 | 82.8 | 41.76 | 0.8367 | 0.0237 | 0.9932 |
| **v4** asc+desc, combined | Occluded | — | — | — | — | — | — |
| **v4** asc+desc, combined | Observed | — | — | — | — | — | — |
| **v4** mix_closest, combined | Occluded | — | — | — | — | — | — |
| **v4** mix_closest, combined | Observed | — | — | — | — | — | — |

## 4. Meilleurs modèles — Métriques Occluded

### Random Fully Masked

#### Classement (pixels occluded, score = rang moyen pondéré)

| # | Modèle | Score | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **v4** asc+desc, combined | 1.00 | 277.0 | 439.9 | 28.52 | 0.5383 | 0.0636 | 0.8378 |
| 2 | **v3** loss (L1+SSIM+L1_occ) | 2.56 | 351.0 | 528.1 | 26.60 | 0.4860 | 0.0861 | 0.7850 |
| 3 | **ALL_SAR_120_epochs** (mix_closest) | 3.22 | 352.7 | 520.1 | 26.70 | 0.4409 | 0.0886 | 0.7902 |
| 4 | **v3** combined (wider+cyclic+loss) | 3.78 | 353.2 | 532.8 | 26.48 | 0.4803 | 0.0869 | 0.7823 |
| 5 | **v2** mix_closest, random_clouds | 5.44 | 360.5 | 538.3 | 26.36 | 0.4566 | 0.0903 | 0.7770 |

#### Détail par bande du meilleur modèle : **v4** asc+desc, combined

| Métrique | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MAE | 180.1 | 188.8 | 233.3 | 239.6 | 309.6 | 346.8 | 372.6 | 349.6 | 291.5 | 258.1 |
| RMSE | 300.1 | 311.4 | 376.6 | 374.0 | 456.5 | 505.6 | 536.9 | 507.0 | 412.0 | 376.9 |
| PSNR | 34.14 | 33.26 | 30.93 | 30.87 | 28.21 | 27.15 | 26.48 | 27.05 | 28.51 | 29.41 |
| SSIM | 0.4503 | 0.4917 | 0.5015 | 0.5455 | 0.5642 | 0.5742 | 0.5551 | 0.5783 | 0.5660 | 0.5561 |
| R2 | 0.5801 | 0.6243 | 0.6607 | 0.6335 | 0.6479 | 0.6828 | 0.6764 | 0.6913 | 0.6780 | 0.7033 |

### Consecutive Fully Masked

#### Classement (pixels occluded, score = rang moyen pondéré)

| # | Modèle | Score | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **v3** combined (wider+cyclic+loss) | 1.00 | 521.7 | 829.7 | 22.99 | 0.3176 | 0.1063 | 0.5993 |
| 2 | **v3** loss (L1+SSIM+L1_occ) | 2.89 | 568.2 | 855.1 | 22.60 | 0.3146 | 0.1093 | 0.5880 |
| 3 | **v2** mix_closest, random_clouds | 3.56 | 542.1 | 847.0 | 22.71 | 0.2766 | 0.1151 | 0.5809 |
| 4 | **v2** asc+desc, random_clouds | 5.44 | 567.2 | 858.5 | 22.49 | 0.2374 | 0.1191 | 0.5722 |
| 5 | asc+desc, random_fully_masked, DA | 6.33 | 698.2 | 1026.0 | 22.28 | 0.2904 | 0.1122 | 0.5836 |

#### Détail par bande du meilleur modèle : **v3** combined (wider+cyclic+loss)

| Métrique | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MAE | 381.2 | 386.0 | 447.1 | 455.7 | 585.6 | 661.6 | 689.6 | 676.3 | 496.8 | 437.4 |
| RMSE | 738.7 | 713.8 | 768.6 | 762.7 | 850.5 | 926.1 | 957.2 | 937.2 | 673.5 | 605.1 |
| PSNR | 26.16 | 26.04 | 24.78 | 24.68 | 22.78 | 21.77 | 21.41 | 21.56 | 23.99 | 24.98 |
| SSIM | 0.2546 | 0.2854 | 0.2900 | 0.3188 | 0.3365 | 0.3412 | 0.3273 | 0.3474 | 0.3429 | 0.3320 |
| R2 | 0.2245 | 0.2743 | 0.3203 | 0.2957 | 0.2780 | 0.3008 | 0.3140 | 0.3107 | 0.3471 | 0.3720 |

## 5. Disponibilité des résultats

| Modèle | Random Fully Masked | Consecutive Fully Masked |
|:---|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | ✅ | ✅ |
| asc+desc, random_clouds, DA | ❌ | ❌ |
| asc+desc, random_clouds | ❌ | ❌ |
| asc+desc, random_fully_masked, DA | ✅ | ✅ |
| mix_closest, random_fully_masked | ❌ | ❌ |
| mix_closest, random_fully_masked, DA | ❌ | ❌ |
| coherence_only | ✅ | ✅ |
| **v2** mix_closest, random_fully_masked | ✅ | ✅ |
| **v2** mix_closest, random_clouds | ✅ | ✅ |
| **v2** asc+desc, random_fully_masked | ✅ | ✅ |
| **v2** asc+desc, random_clouds | ✅ | ✅ |
| **v3** mix_closest, random_clouds | ✅ | ✅ |
| **v3** loss (L1+SSIM+L1_occ) | ✅ | ✅ |
| **v3** wider | ✅ | ✅ |
| **v3** combined (wider+cyclic+loss) | ✅ | ✅ |
| **v3** cyclic | ✅ | ✅ |
| **v4** asc+desc, combined | ✅ | ❌ |
| **v4** mix_closest, combined | ✅ | ❌ |
| **v4** asc+desc, combined, NDVI+R² | ❌ | ❌ |
| **v4** mix_closest, combined, NDVI+R² | ❌ | ❌ |

## 6. Métriques à la Parcelle (RPG)

### Masquage : Random Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Global | — | — | — | — | — | — |
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | — | — | — | — | — | — |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | — | — | — | — | — | — |
| **v2** mix_closest, random_clouds | Global | 108.8 | 208.3 | 35.62 | 0.7326 | 0.0351 | 0.9526 |
| **v2** mix_closest, random_clouds | Occluded | 368.0 | 549.8 | 26.16 | 0.4566 | 0.0916 | 0.7763 |
| **v2** mix_closest, random_clouds | Observed | 59.2 | 84.5 | 41.61 | 0.8306 | 0.0242 | 0.9931 |
| **v3** combined (wider+cyclic+loss) | Global | 115.5 | 214.5 | 35.08 | 0.7377 | 0.0370 | 0.9510 |
| **v3** combined (wider+cyclic+loss) | Occluded | 360.9 | 543.1 | 26.30 | 0.4803 | 0.0884 | 0.7817 |
| **v3** combined (wider+cyclic+loss) | Observed | 68.6 | 98.3 | 40.29 | 0.8307 | 0.0272 | 0.9900 |

### Masquage : Consecutive Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Global | — | — | — | — | — | — |
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | — | — | — | — | — | — |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | — | — | — | — | — | — |
| **v2** mix_closest, random_clouds | Global | — | — | — | — | — | — |
| **v2** mix_closest, random_clouds | Occluded | — | — | — | — | — | — |
| **v2** mix_closest, random_clouds | Observed | — | — | — | — | — | — |
| **v3** combined (wider+cyclic+loss) | Global | 123.2 | 266.6 | 33.85 | 0.6901 | 0.0370 | 0.9267 |
| **v3** combined (wider+cyclic+loss) | Occluded | 524.4 | 831.9 | 22.98 | 0.3176 | 0.1059 | 0.6088 |
| **v3** combined (wider+cyclic+loss) | Observed | 69.0 | 98.9 | 40.24 | 0.8299 | 0.0272 | 0.9900 |

## 7. Impact du Filtrage des Pixels Noirs (nodata)

Comparaison des métriques **avant** et **après** exclusion des pixels où toutes les bandes = 0 dans la cible.

| Modèle | Masquage | Version | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Random Fully Masked | Sans filtrage | — | — | — | — | — | — |
| **ALL_SAR_120_epochs** (mix_closest) | Random Fully Masked | Avec filtrage | 352.7 | 520.1 | 26.70 | 0.4409 | 0.0886 | 0.7902 |
| **ALL_SAR_120_epochs** (mix_closest) | Consecutive Fully Masked | Sans filtrage | — | — | — | — | — | — |
| **ALL_SAR_120_epochs** (mix_closest) | Consecutive Fully Masked | Avec filtrage | 594.2 | 880.5 | 22.20 | 0.2190 | 0.1284 | 0.5547 |
