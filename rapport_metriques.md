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

## 4. Métriques par bande spectrale

### SSIM par bande

#### Masquage : Random Fully Masked

| Modèle | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 0.5482 | 0.6355 | 0.6518 | 0.7183 | 0.7721 | 0.7923 | 0.7425 | 0.7917 | 0.7488 | 0.7274 |
| asc+desc, random_clouds, DA | 0.6030 | 0.6742 | 0.6824 | 0.7641 | 0.8065 | 0.8238 | 0.7683 | 0.8239 | 0.7958 | 0.7724 |
| asc+desc, random_clouds | 0.6465 | 0.7092 | 0.7203 | 0.7751 | 0.8118 | 0.8299 | 0.8036 | 0.8282 | 0.8044 | 0.7825 |
| asc+desc, random_fully_masked, DA | 0.5688 | 0.6508 | 0.6544 | 0.7270 | 0.7681 | 0.7863 | 0.7510 | 0.7888 | 0.7631 | 0.7442 |
| mix_closest, random_fully_masked | 0.5599 | 0.6461 | 0.6525 | 0.7181 | 0.7644 | 0.7838 | 0.7468 | 0.7834 | 0.7687 | 0.7437 |
| mix_closest, random_fully_masked, DA | 0.5957 | 0.6732 | 0.6804 | 0.7427 | 0.7731 | 0.7946 | 0.7687 | 0.7915 | 0.7780 | 0.7596 |
| coherence_only | 0.5442 | 0.6173 | 0.6230 | 0.7065 | 0.7719 | 0.7916 | 0.7376 | 0.7951 | 0.7420 | 0.7191 |
| **v2** mix_closest, random_fully_masked | 0.5774 | 0.6585 | 0.6678 | 0.7288 | 0.7734 | 0.7890 | 0.7387 | 0.7908 | 0.7675 | 0.7420 |
| **v2** mix_closest, random_clouds | 0.5842 | 0.6628 | 0.6713 | 0.7335 | 0.7743 | 0.7943 | 0.7690 | 0.7974 | 0.7811 | 0.7580 |
| **v2** asc+desc, random_fully_masked | 0.5678 | 0.6434 | 0.6491 | 0.7279 | 0.7717 | 0.7879 | 0.7441 | 0.7898 | 0.7709 | 0.7477 |
| **v2** asc+desc, random_clouds | 0.5587 | 0.6341 | 0.6423 | 0.7142 | 0.7663 | 0.7840 | 0.7273 | 0.7846 | 0.7495 | 0.7301 |

#### Masquage : Consecutive Fully Masked

| Modèle | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 0.5055 | 0.5826 | 0.5994 | 0.6569 | 0.7083 | 0.7285 | 0.6835 | 0.7273 | 0.6881 | 0.6678 |
| asc+desc, random_clouds, DA | 0.5554 | 0.6186 | 0.6281 | 0.6980 | 0.7363 | 0.7528 | 0.7033 | 0.7521 | 0.7281 | 0.7087 |
| asc+desc, random_clouds | 0.5994 | 0.6545 | 0.6665 | 0.7108 | 0.7423 | 0.7595 | 0.7379 | 0.7569 | 0.7384 | 0.7200 |
| asc+desc, random_fully_masked, DA | 0.5407 | 0.6169 | 0.6228 | 0.6859 | 0.7279 | 0.7461 | 0.7141 | 0.7472 | 0.7246 | 0.7061 |
| mix_closest, random_fully_masked | 0.5298 | 0.6106 | 0.6189 | 0.6737 | 0.7221 | 0.7415 | 0.7086 | 0.7403 | 0.7242 | 0.7021 |
| mix_closest, random_fully_masked, DA | 0.5520 | 0.6249 | 0.6344 | 0.6876 | 0.7230 | 0.7455 | 0.7217 | 0.7414 | 0.7230 | 0.7078 |
| coherence_only | 0.5041 | 0.5698 | 0.5769 | 0.6494 | 0.7073 | 0.7260 | 0.6789 | 0.7282 | 0.6827 | 0.6643 |
| **v2** mix_closest, random_fully_masked | 0.5387 | 0.6136 | 0.6260 | 0.6801 | 0.7267 | 0.7429 | 0.6954 | 0.7442 | 0.7194 | 0.6962 |
| **v2** mix_closest, random_clouds | 0.5415 | 0.6141 | 0.6252 | 0.6795 | 0.7219 | 0.7421 | 0.7185 | 0.7446 | 0.7291 | 0.7078 |
| **v2** asc+desc, random_fully_masked | 0.5310 | 0.6009 | 0.6096 | 0.6804 | 0.7251 | 0.7424 | 0.7015 | 0.7430 | 0.7232 | 0.7029 |
| **v2** asc+desc, random_clouds | 0.5176 | 0.5865 | 0.5984 | 0.6581 | 0.7114 | 0.7295 | 0.6784 | 0.7302 | 0.6958 | 0.6799 |

### PSNR par bande

#### Masquage : Random Fully Masked

| Modèle | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 35.06 | 35.77 | 33.97 | 33.76 | 31.65 | 31.43 | 29.28 | 30.87 | 32.36 | 32.32 |
| asc+desc, random_clouds, DA | 34.46 | 34.39 | 32.76 | 32.99 | 30.22 | 29.85 | 27.67 | 29.22 | 30.95 | 31.12 |
| asc+desc, random_clouds | 35.51 | 34.84 | 33.39 | 33.45 | 30.30 | 29.91 | 28.00 | 29.27 | 31.08 | 31.46 |
| asc+desc, random_fully_masked, DA | 34.53 | 35.53 | 33.33 | 33.43 | 30.97 | 30.64 | 28.76 | 30.10 | 32.23 | 32.08 |
| mix_closest, random_fully_masked | 34.17 | 35.18 | 32.95 | 33.01 | 30.67 | 30.29 | 28.58 | 29.82 | 31.83 | 31.84 |
| mix_closest, random_fully_masked, DA | 34.67 | 35.39 | 33.37 | 34.09 | 30.72 | 30.25 | 28.56 | 29.68 | 31.72 | 32.40 |
| coherence_only | 33.87 | 34.31 | 32.47 | 32.24 | 30.00 | 29.69 | 27.70 | 29.14 | 30.42 | 30.79 |
| **v2** mix_closest, random_fully_masked | 34.26 | 35.33 | 33.18 | 33.33 | 30.77 | 30.20 | 28.37 | 29.74 | 31.82 | 31.70 |
| **v2** mix_closest, random_clouds | 35.71 | 36.15 | 34.27 | 34.64 | 31.80 | 31.31 | 29.45 | 30.83 | 32.80 | 33.35 |
| **v2** asc+desc, random_fully_masked | 34.39 | 35.30 | 33.11 | 33.26 | 30.80 | 30.28 | 28.52 | 29.87 | 31.86 | 31.93 |
| **v2** asc+desc, random_clouds | 35.40 | 35.81 | 33.53 | 34.03 | 31.81 | 31.00 | 28.98 | 30.64 | 32.29 | 32.27 |

#### Masquage : Consecutive Fully Masked

| Modèle | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 32.68 | 33.40 | 32.22 | 31.86 | 30.19 | 30.04 | 28.08 | 29.44 | 31.22 | 31.33 |
| asc+desc, random_clouds, DA | 29.70 | 29.59 | 28.78 | 28.41 | 26.04 | 25.89 | 23.90 | 25.19 | 26.85 | 27.59 |
| asc+desc, random_clouds | 30.62 | 30.08 | 29.43 | 28.97 | 26.28 | 26.11 | 24.35 | 25.41 | 27.10 | 27.96 |
| asc+desc, random_fully_masked, DA | 32.45 | 33.60 | 32.13 | 32.01 | 30.26 | 30.19 | 28.35 | 29.60 | 31.70 | 31.76 |
| mix_closest, random_fully_masked | 31.80 | 32.84 | 31.48 | 31.17 | 29.47 | 29.43 | 27.84 | 28.99 | 30.99 | 31.23 |
| mix_closest, random_fully_masked, DA | 31.45 | 32.21 | 30.95 | 31.39 | 29.19 | 29.03 | 27.44 | 28.44 | 29.57 | 30.87 |
| coherence_only | 29.13 | 29.34 | 28.41 | 27.64 | 25.67 | 25.59 | 23.80 | 24.99 | 26.29 | 27.22 |
| **v2** mix_closest, random_fully_masked | 32.14 | 33.29 | 31.83 | 31.77 | 30.05 | 29.73 | 27.95 | 29.23 | 31.19 | 31.29 |
| **v2** mix_closest, random_clouds | 33.48 | 34.07 | 32.86 | 33.02 | 30.68 | 30.39 | 28.63 | 29.85 | 32.02 | 32.79 |
| **v2** asc+desc, random_fully_masked | 32.12 | 33.15 | 31.71 | 31.58 | 29.97 | 29.75 | 28.03 | 29.30 | 30.96 | 31.22 |
| **v2** asc+desc, random_clouds | 33.21 | 33.72 | 32.16 | 32.38 | 30.75 | 30.17 | 28.24 | 29.75 | 31.48 | 31.72 |

### MAE par bande

#### Masquage : Random Fully Masked

| Modèle | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 198.8 | 203.5 | 221.7 | 259.0 | 354.6 | 380.7 | 432.1 | 407.5 | 323.4 | 271.4 |
| asc+desc, random_clouds, DA | 159.2 | 165.7 | 184.5 | 200.6 | 283.0 | 304.8 | 352.0 | 326.4 | 261.0 | 225.7 |
| asc+desc, random_clouds | 142.9 | 160.0 | 174.2 | 193.5 | 280.2 | 301.5 | 341.7 | 323.5 | 259.7 | 219.1 |
| asc+desc, random_fully_masked, DA | 228.3 | 227.0 | 251.6 | 281.5 | 386.4 | 413.8 | 461.8 | 441.1 | 340.0 | 297.0 |
| mix_closest, random_fully_masked | 265.6 | 266.0 | 292.8 | 327.7 | 432.3 | 461.4 | 503.6 | 482.5 | 384.3 | 335.6 |
| mix_closest, random_fully_masked, DA | 283.4 | 287.3 | 315.3 | 336.5 | 425.3 | 459.0 | 516.9 | 495.4 | 427.8 | 347.8 |
| coherence_only | 165.6 | 161.0 | 184.1 | 204.9 | 278.1 | 296.9 | 342.3 | 317.5 | 262.9 | 225.7 |
| **v2** mix_closest, random_fully_masked | 240.0 | 234.5 | 262.0 | 287.4 | 387.7 | 422.6 | 470.4 | 445.9 | 354.1 | 313.2 |
| **v2** mix_closest, random_clouds | 184.8 | 191.9 | 210.6 | 238.2 | 348.9 | 379.5 | 422.8 | 401.6 | 311.5 | 255.7 |
| **v2** asc+desc, random_fully_masked | 232.7 | 231.5 | 260.8 | 292.1 | 391.9 | 424.1 | 470.0 | 448.8 | 367.8 | 315.6 |
| **v2** asc+desc, random_clouds | 194.7 | 201.3 | 231.4 | 252.8 | 343.6 | 381.0 | 427.3 | 402.0 | 323.5 | 278.5 |

#### Masquage : Consecutive Fully Masked

| Modèle | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 213.9 | 217.7 | 235.3 | 273.9 | 370.6 | 397.8 | 450.4 | 426.7 | 334.8 | 282.7 |
| asc+desc, random_clouds, DA | 242.5 | 257.6 | 269.6 | 306.9 | 419.6 | 449.9 | 502.3 | 481.8 | 389.9 | 321.5 |
| asc+desc, random_clouds | 224.1 | 249.8 | 257.0 | 297.0 | 415.3 | 444.6 | 488.7 | 475.6 | 384.1 | 313.9 |
| asc+desc, random_fully_masked, DA | 243.4 | 239.8 | 260.8 | 293.1 | 392.3 | 414.7 | 463.1 | 443.0 | 344.6 | 299.5 |
| mix_closest, random_fully_masked | 285.6 | 285.4 | 307.4 | 347.0 | 448.4 | 471.3 | 512.8 | 491.5 | 393.1 | 343.5 |
| mix_closest, random_fully_masked, DA | 321.4 | 324.7 | 351.8 | 373.8 | 451.2 | 480.1 | 539.6 | 518.3 | 474.1 | 376.3 |
| coherence_only | 255.1 | 259.8 | 275.2 | 319.7 | 427.5 | 455.3 | 504.8 | 485.5 | 399.1 | 328.0 |
| **v2** mix_closest, random_fully_masked | 256.0 | 248.9 | 274.0 | 301.2 | 394.5 | 424.8 | 473.1 | 448.9 | 361.1 | 318.2 |
| **v2** mix_closest, random_clouds | 194.6 | 200.9 | 218.1 | 247.1 | 358.1 | 387.1 | 430.6 | 411.0 | 316.9 | 258.0 |
| **v2** asc+desc, random_fully_masked | 250.4 | 246.9 | 273.1 | 307.1 | 399.5 | 426.2 | 472.9 | 451.9 | 376.7 | 322.9 |
| **v2** asc+desc, random_clouds | 205.4 | 212.2 | 240.8 | 263.8 | 351.9 | 387.2 | 434.2 | 409.6 | 329.4 | 282.7 |

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
