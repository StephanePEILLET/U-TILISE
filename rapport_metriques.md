# Rapport de Métriques — Cloud Reconstruction U-TILISE

**Source des résultats :** logs SLURM dans `/mnt/stores/store_dai/tmp/speillet/logs`

## 1. Récapitulatif Global

### Masquage : Random Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 305.3 | 466.4 | 31.89 | 0.7129 | 0.0385 | 0.8487 |
| asc+desc, random_clouds, DA | 246.3 | 513.4 | 30.48 | 0.7515 | 0.0664 | 0.7896 |
| asc+desc, random_clouds | 239.6 | 512.5 | 30.83 | 0.7711 | 0.0695 | 0.7869 |
| asc+desc, random_fully_masked, DA | 332.9 | 534.7 | 31.30 | 0.7202 | 0.0404 | 0.8343 |
| mix_closest, random_fully_masked | 375.2 | 585.6 | 31.01 | 0.7167 | 0.0416 | 0.8306 |
| mix_closest, random_fully_masked, DA | 389.5 | 610.2 | 31.21 | 0.7357 | 0.0382 | 0.8353 |
| coherence_only | 243.9 | 504.7 | 30.23 | 0.7048 | 0.0696 | 0.7935 |
| **v2** mix_closest, random_fully_masked | 341.8 | 551.4 | 30.99 | 0.7234 | 0.0417 | 0.8305 |
| **v2** mix_closest, random_clouds | 294.5 | 462.0 | 32.20 | 0.7326 | 0.0353 | 0.8473 |
| **v2** asc+desc, random_fully_masked | 343.5 | 548.8 | 31.08 | 0.7200 | 0.0408 | 0.8313 |
| **v2** asc+desc, random_clouds | 303.6 | 469.4 | 31.73 | 0.7091 | 0.0392 | 0.8433 |

### Masquage : Consecutive Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 320.4 | 524.9 | 30.39 | 0.6548 | 0.0411 | 0.8197 |
| asc+desc, random_clouds, DA | 364.2 | 819.4 | 26.51 | 0.6881 | 0.0987 | 0.6286 |
| asc+desc, random_clouds | 355.0 | 810.5 | 26.95 | 0.7086 | 0.1284 | 0.6326 |
| asc+desc, random_fully_masked, DA | 339.4 | 576.2 | 30.56 | 0.6832 | 0.0395 | 0.8166 |
| mix_closest, random_fully_masked | 388.6 | 640.2 | 29.93 | 0.6772 | 0.0406 | 0.8054 |
| mix_closest, random_fully_masked, DA | 421.1 | 709.6 | 29.47 | 0.6861 | 0.0399 | 0.7864 |
| coherence_only | 371.0 | 826.0 | 26.16 | 0.6488 | 0.1002 | 0.6264 |
| **v2** mix_closest, random_fully_masked | 350.0 | 595.5 | 30.21 | 0.6783 | 0.0412 | 0.8110 |
| **v2** mix_closest, random_clouds | 302.2 | 508.3 | 31.09 | 0.6824 | 0.0360 | 0.8243 |
| **v2** asc+desc, random_fully_masked | 352.8 | 597.6 | 30.17 | 0.6760 | 0.0403 | 0.8087 |
| **v2** asc+desc, random_clouds | 311.7 | 513.8 | 30.69 | 0.6586 | 0.0393 | 0.8215 |

## 2. Métriques sur pixels reconstruits (Occluded)

### Masquage : Random Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 899.0 | 1108.3 | 22.54 | 0.4409 | 0.0886 | 0.6645 |
| asc+desc, random_clouds, DA | 998.2 | 1336.7 | 23.21 | 0.4328 | 0.2314 | 0.3926 |
| asc+desc, random_clouds | 983.2 | 1336.8 | 23.72 | 0.4275 | 0.2681 | 0.3750 |
| asc+desc, random_fully_masked, DA | 1055.5 | 1325.0 | 21.31 | 0.4019 | 0.1014 | 0.6231 |
| mix_closest, random_fully_masked | 1140.2 | 1412.7 | 21.09 | 0.4173 | 0.1036 | 0.6235 |
| mix_closest, random_fully_masked, DA | 1151.2 | 1440.3 | 21.14 | 0.4576 | 0.1010 | 0.6283 |
| coherence_only | 984.0 | 1314.7 | 22.04 | 0.3466 | 0.2360 | 0.4000 |
| **v2** mix_closest, random_fully_masked | 1085.4 | 1361.0 | 21.32 | 0.4587 | 0.0976 | 0.6275 |
| **v2** mix_closest, random_clouds | 900.9 | 1120.3 | 22.26 | 0.4566 | 0.0903 | 0.6500 |
| **v2** asc+desc, random_fully_masked | 1070.0 | 1343.3 | 21.33 | 0.4591 | 0.0980 | 0.6270 |
| **v2** asc+desc, random_clouds | 916.5 | 1128.8 | 22.04 | 0.4229 | 0.0965 | 0.6420 |

### Masquage : Consecutive Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 1081.9 | 1366.0 | 19.35 | 0.2190 | 0.1284 | 0.4716 |
| asc+desc, random_clouds, DA | 2296.9 | 2595.1 | 16.45 | 0.1560 | 0.6064 | 0.0721 |
| asc+desc, random_clouds | 2266.2 | 2570.1 | 17.09 | 0.1581 | 0.8594 | 0.0573 |
| asc+desc, random_fully_masked, DA | 1157.0 | 1512.9 | 19.39 | 0.2904 | 0.1122 | 0.5013 |
| mix_closest, random_fully_masked | 1305.4 | 1650.9 | 18.62 | 0.2937 | 0.1147 | 0.4997 |
| mix_closest, random_fully_masked, DA | 1437.6 | 1822.1 | 17.70 | 0.2879 | 0.1353 | 0.4253 |
| coherence_only | 2309.3 | 2608.6 | 15.12 | 0.1041 | 0.5966 | 0.1403 |
| **v2** mix_closest, random_fully_masked | 1195.6 | 1559.4 | 19.31 | 0.3089 | 0.1106 | 0.5010 |
| **v2** mix_closest, random_clouds | 1032.5 | 1335.7 | 19.74 | 0.2766 | 0.1151 | 0.4927 |
| **v2** asc+desc, random_fully_masked | 1192.1 | 1559.4 | 19.13 | 0.3149 | 0.1111 | 0.4921 |
| **v2** asc+desc, random_clouds | 1055.4 | 1341.4 | 19.59 | 0.2374 | 0.1191 | 0.4830 |

## 3. Occluded vs Observed (détail)

### Masquage : Random Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | 899.0 | 1108.3 | 22.54 | 0.4409 | 0.0886 | 0.6645 |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | 69.2 | 99.0 | 40.19 | 0.8106 | 0.0288 | 0.9911 |
| asc+desc, random_clouds, DA | Occluded | 998.2 | 1336.7 | 23.21 | 0.4328 | 0.2314 | 0.3926 |
| asc+desc, random_clouds, DA | Observed | 64.3 | 93.8 | 40.71 | 0.8186 | 0.0271 | 0.9906 |
| asc+desc, random_clouds | Occluded | 983.2 | 1336.8 | 23.72 | 0.4275 | 0.2681 | 0.3750 |
| asc+desc, random_clouds | Observed | 56.1 | 82.1 | 41.83 | 0.8450 | 0.0234 | 0.9932 |
| asc+desc, random_fully_masked, DA | Occluded | 1055.5 | 1325.0 | 21.31 | 0.4019 | 0.1014 | 0.6231 |
| asc+desc, random_fully_masked, DA | Observed | 67.7 | 98.3 | 40.33 | 0.8276 | 0.0290 | 0.9891 |
| mix_closest, random_fully_masked | Occluded | 1140.2 | 1412.7 | 21.09 | 0.4173 | 0.1036 | 0.6235 |
| mix_closest, random_fully_masked | Observed | 71.1 | 102.9 | 39.94 | 0.8191 | 0.0300 | 0.9877 |
| mix_closest, random_fully_masked, DA | Occluded | 1151.2 | 1440.3 | 21.14 | 0.4576 | 0.1010 | 0.6283 |
| mix_closest, random_fully_masked, DA | Observed | 64.7 | 94.5 | 40.64 | 0.8339 | 0.0263 | 0.9912 |
| coherence_only | Occluded | 984.0 | 1314.7 | 22.04 | 0.3466 | 0.2360 | 0.4000 |
| coherence_only | Observed | 71.3 | 103.8 | 39.83 | 0.8044 | 0.0299 | 0.9883 |
| **v2** mix_closest, random_fully_masked | Occluded | 1085.4 | 1361.0 | 21.32 | 0.4587 | 0.0976 | 0.6275 |
| **v2** mix_closest, random_fully_masked | Observed | 73.3 | 106.3 | 39.63 | 0.8173 | 0.0311 | 0.9867 |
| **v2** mix_closest, random_clouds | Occluded | 900.9 | 1120.3 | 22.26 | 0.4566 | 0.0903 | 0.6500 |
| **v2** mix_closest, random_clouds | Observed | 59.7 | 85.5 | 41.51 | 0.8306 | 0.0249 | 0.9922 |
| **v2** asc+desc, random_fully_masked | Occluded | 1070.0 | 1343.3 | 21.33 | 0.4591 | 0.0980 | 0.6270 |
| **v2** asc+desc, random_fully_masked | Observed | 71.5 | 103.5 | 39.87 | 0.8123 | 0.0301 | 0.9875 |
| **v2** asc+desc, random_clouds | Occluded | 916.5 | 1128.8 | 22.04 | 0.4229 | 0.0965 | 0.6420 |
| **v2** asc+desc, random_clouds | Observed | 66.2 | 96.1 | 40.52 | 0.8095 | 0.0282 | 0.9894 |

### Masquage : Consecutive Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | 1081.9 | 1366.0 | 19.35 | 0.2190 | 0.1284 | 0.4716 |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | 69.4 | 99.4 | 40.16 | 0.8098 | 0.0289 | 0.9910 |
| asc+desc, random_clouds, DA | Occluded | 2296.9 | 2595.1 | 16.45 | 0.1560 | 0.6064 | 0.0721 |
| asc+desc, random_clouds, DA | Observed | 64.5 | 94.2 | 40.67 | 0.8180 | 0.0271 | 0.9905 |
| asc+desc, random_clouds | Occluded | 2266.2 | 2570.1 | 17.09 | 0.1581 | 0.8594 | 0.0573 |
| asc+desc, random_clouds | Observed | 56.3 | 82.3 | 41.81 | 0.8445 | 0.0234 | 0.9932 |
| asc+desc, random_fully_masked, DA | Occluded | 1157.0 | 1512.9 | 19.39 | 0.2904 | 0.1122 | 0.5013 |
| asc+desc, random_fully_masked, DA | Observed | 67.9 | 98.5 | 40.31 | 0.8272 | 0.0290 | 0.9890 |
| mix_closest, random_fully_masked | Occluded | 1305.4 | 1650.9 | 18.62 | 0.2937 | 0.1147 | 0.4997 |
| mix_closest, random_fully_masked | Observed | 71.3 | 103.1 | 39.92 | 0.8185 | 0.0300 | 0.9877 |
| mix_closest, random_fully_masked, DA | Occluded | 1437.6 | 1822.1 | 17.70 | 0.2879 | 0.1353 | 0.4253 |
| mix_closest, random_fully_masked, DA | Observed | 65.1 | 94.9 | 40.60 | 0.8332 | 0.0264 | 0.9912 |
| coherence_only | Occluded | 2309.3 | 2608.6 | 15.12 | 0.1041 | 0.5966 | 0.1403 |
| coherence_only | Observed | 71.6 | 104.2 | 39.79 | 0.8038 | 0.0300 | 0.9883 |
| **v2** mix_closest, random_fully_masked | Occluded | 1195.6 | 1559.4 | 19.31 | 0.3089 | 0.1106 | 0.5010 |
| **v2** mix_closest, random_fully_masked | Observed | 73.7 | 106.7 | 39.60 | 0.8166 | 0.0311 | 0.9867 |
| **v2** mix_closest, random_clouds | Occluded | 1032.5 | 1335.7 | 19.74 | 0.2766 | 0.1151 | 0.4927 |
| **v2** mix_closest, random_clouds | Observed | 59.9 | 85.8 | 41.47 | 0.8300 | 0.0249 | 0.9922 |
| **v2** asc+desc, random_fully_masked | Occluded | 1192.1 | 1559.4 | 19.13 | 0.3149 | 0.1111 | 0.4921 |
| **v2** asc+desc, random_fully_masked | Observed | 71.9 | 104.1 | 39.82 | 0.8117 | 0.0302 | 0.9875 |
| **v2** asc+desc, random_clouds | Occluded | 1055.4 | 1341.4 | 19.59 | 0.2374 | 0.1191 | 0.4830 |
| **v2** asc+desc, random_clouds | Observed | 66.4 | 96.4 | 40.48 | 0.8088 | 0.0283 | 0.9893 |

## 4. Meilleurs modèles — Métriques Occluded

### Random Fully Masked

#### Classement (pixels occluded, score = rang moyen pondéré)

| # | Modèle | Score | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **ALL_SAR_120_epochs** (mix_closest) | 1.67 | 899.0 | 1108.3 | 22.54 | 0.4409 | 0.0886 | 0.6645 |
| 2 | **v2** mix_closest, random_clouds | 2.44 | 900.9 | 1120.3 | 22.26 | 0.4566 | 0.0903 | 0.6500 |
| 3 | **v2** asc+desc, random_clouds | 3.89 | 916.5 | 1128.8 | 22.04 | 0.4229 | 0.0965 | 0.6420 |
| 4 | **v2** asc+desc, random_fully_masked | 6.33 | 1070.0 | 1343.3 | 21.33 | 0.4591 | 0.0980 | 0.6270 |
| 5 | **v2** mix_closest, random_fully_masked | 6.67 | 1085.4 | 1361.0 | 21.32 | 0.4587 | 0.0976 | 0.6275 |

#### Détail par bande du meilleur modèle : **ALL_SAR_120_epochs** (mix_closest)

| Métrique | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MAE | 517.3 | 595.0 | 634.3 | 743.6 | 1050.9 | 1186.0 | 1252.1 | 1248.4 | 988.8 | 774.0 |
| RMSE | 635.4 | 718.7 | 771.0 | 880.8 | 1218.5 | 1373.3 | 1445.0 | 1440.2 | 1134.7 | 904.3 |
| PSNR | 28.25 | 27.17 | 25.48 | 25.04 | 22.04 | 20.80 | 20.23 | 20.54 | 22.50 | 23.82 |
| SSIM | 0.3287 | 0.3831 | 0.3937 | 0.4551 | 0.4868 | 0.4919 | 0.4484 | 0.5003 | 0.4670 | 0.4536 |
| R2 | 0.3984 | 0.4447 | 0.4628 | 0.4506 | 0.4513 | 0.4715 | 0.4661 | 0.4844 | 0.4648 | 0.4900 |

### Consecutive Fully Masked

#### Classement (pixels occluded, score = rang moyen pondéré)

| # | Modèle | Score | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **v2** mix_closest, random_clouds | 2.67 | 1032.5 | 1335.7 | 19.74 | 0.2766 | 0.1151 | 0.4927 |
| 2 | asc+desc, random_fully_masked, DA | 3.11 | 1157.0 | 1512.9 | 19.39 | 0.2904 | 0.1122 | 0.5013 |
| 3 | **v2** mix_closest, random_fully_masked | 3.78 | 1195.6 | 1559.4 | 19.31 | 0.3089 | 0.1106 | 0.5010 |
| 4 | **v2** asc+desc, random_clouds | 3.89 | 1055.4 | 1341.4 | 19.59 | 0.2374 | 0.1191 | 0.4830 |
| 5 | **v2** asc+desc, random_fully_masked | 4.56 | 1192.1 | 1559.4 | 19.13 | 0.3149 | 0.1111 | 0.4921 |

#### Détail par bande du meilleur modèle : **v2** mix_closest, random_clouds

| Métrique | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MAE | 630.1 | 700.5 | 737.2 | 851.7 | 1223.0 | 1368.7 | 1427.0 | 1427.3 | 1089.5 | 869.8 |
| RMSE | 911.9 | 961.9 | 995.1 | 1103.7 | 1466.8 | 1621.0 | 1683.3 | 1683.1 | 1270.1 | 1026.6 |
| PSNR | 23.62 | 23.15 | 22.31 | 21.72 | 19.17 | 18.20 | 17.82 | 17.93 | 20.32 | 21.79 |
| SSIM | 0.2018 | 0.2304 | 0.2403 | 0.2770 | 0.2991 | 0.3060 | 0.2901 | 0.3136 | 0.3112 | 0.2961 |
| R2 | 0.1798 | 0.2209 | 0.2615 | 0.2437 | 0.2127 | 0.2307 | 0.2398 | 0.2391 | 0.2838 | 0.3059 |

## 5. Disponibilité des résultats

| Modèle | Random Fully Masked | Consecutive Fully Masked |
|:---|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | ✅ | ✅ |
| asc+desc, random_clouds, DA | ✅ | ✅ |
| asc+desc, random_clouds | ✅ | ✅ |
| asc+desc, random_fully_masked, DA | ✅ | ✅ |
| mix_closest, random_fully_masked | ✅ | ✅ |
| mix_closest, random_fully_masked, DA | ✅ | ✅ |
| coherence_only | ✅ | ✅ |
| **v2** mix_closest, random_fully_masked | ✅ | ✅ |
| **v2** mix_closest, random_clouds | ✅ | ✅ |
| **v2** asc+desc, random_fully_masked | ✅ | ✅ |
| **v2** asc+desc, random_clouds | ✅ | ✅ |
