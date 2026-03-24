# Rapport de Métriques — Cloud Reconstruction U-TILISE

**Source des résultats :** logs SLURM dans `/mnt/stores/store_dai/tmp/speillet/logs`

## 1. Récapitulatif Global

### Masquage : Random Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 1669.4 | 2742.7 | 18.55 | 0.6394 | 0.0735 | 0.3991 |
| asc+desc, random_clouds, DA | 246.3 | 513.4 | 30.48 | 0.7515 | 0.0664 | 0.7896 |
| asc+desc, random_clouds | 239.6 | 512.5 | 30.83 | 0.7711 | 0.0695 | 0.7869 |
| asc+desc, random_fully_masked, DA | 332.9 | 534.7 | 31.30 | 0.7202 | 0.0404 | 0.8343 |
| mix_closest, random_fully_masked | 375.2 | 585.6 | 31.01 | 0.7167 | 0.0416 | 0.8306 |
| mix_closest, random_fully_masked, DA | 389.5 | 610.2 | 31.21 | 0.7357 | 0.0382 | 0.8353 |
| coherence_only | 243.9 | 504.7 | 30.23 | 0.7048 | 0.0696 | 0.7935 |

### Masquage : Consecutive Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 1406.1 | 2486.9 | 19.01 | 0.6275 | 0.0598 | 0.4132 |
| asc+desc, random_clouds, DA | 364.2 | 819.4 | 26.51 | 0.6881 | 0.0987 | 0.6286 |
| asc+desc, random_clouds | 355.0 | 810.5 | 26.95 | 0.7086 | 0.1284 | 0.6326 |
| asc+desc, random_fully_masked, DA | 339.4 | 576.2 | 30.56 | 0.6832 | 0.0395 | 0.8166 |
| mix_closest, random_fully_masked | 388.6 | 640.2 | 29.93 | 0.6772 | 0.0406 | 0.8054 |
| mix_closest, random_fully_masked, DA | 421.1 | 709.6 | 29.47 | 0.6861 | 0.0399 | 0.7864 |
| coherence_only | 371.0 | 826.0 | 26.16 | 0.6488 | 0.1002 | 0.6264 |

## 2. Métriques sur pixels reconstruits (Occluded)

### Masquage : Random Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 6938.1 | 7236.0 | 3.13 | 0.0806 | 0.2969 | 0.0242 |
| asc+desc, random_clouds, DA | 998.2 | 1336.7 | 23.21 | 0.4328 | 0.2314 | 0.3926 |
| asc+desc, random_clouds | 983.2 | 1336.8 | 23.72 | 0.4275 | 0.2681 | 0.3750 |
| asc+desc, random_fully_masked, DA | 1055.5 | 1325.0 | 21.31 | 0.4019 | 0.1014 | 0.6231 |
| mix_closest, random_fully_masked | 1140.2 | 1412.7 | 21.09 | 0.4173 | 0.1036 | 0.6235 |
| mix_closest, random_fully_masked, DA | 1151.2 | 1440.3 | 21.14 | 0.4576 | 0.1010 | 0.6283 |
| coherence_only | 984.0 | 1314.7 | 22.04 | 0.3466 | 0.2360 | 0.4000 |

### Masquage : Consecutive Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 7029.8 | 7273.2 | 3.05 | 0.0820 | 0.2818 | 0.0303 |
| asc+desc, random_clouds, DA | 2296.9 | 2595.1 | 16.45 | 0.1560 | 0.6064 | 0.0721 |
| asc+desc, random_clouds | 2266.2 | 2570.1 | 17.09 | 0.1581 | 0.8594 | 0.0573 |
| asc+desc, random_fully_masked, DA | 1157.0 | 1512.9 | 19.39 | 0.2904 | 0.1122 | 0.5013 |
| mix_closest, random_fully_masked | 1305.4 | 1650.9 | 18.62 | 0.2937 | 0.1147 | 0.4997 |
| mix_closest, random_fully_masked, DA | 1437.6 | 1822.1 | 17.70 | 0.2879 | 0.1353 | 0.4253 |
| coherence_only | 2309.3 | 2608.6 | 15.12 | 0.1041 | 0.5966 | 0.1403 |

## 3. Occluded vs Observed (détail)

### Masquage : Random Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | 6938.1 | 7236.0 | 3.13 | 0.0806 | 0.2969 | 0.0242 |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | 69.9 | 100.2 | 40.08 | 0.8085 | 0.0290 | 0.9908 |
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

### Masquage : Consecutive Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | 7029.8 | 7273.2 | 3.05 | 0.0820 | 0.2818 | 0.0303 |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | 70.1 | 100.6 | 40.05 | 0.8078 | 0.0291 | 0.9907 |
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

## 4. Métriques par bande spectrale

### SSIM par bande

#### Masquage : Random Fully Masked

| Modèle | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 0.4925 | 0.5707 | 0.5851 | 0.6416 | 0.6909 | 0.7105 | 0.6683 | 0.7086 | 0.6724 | 0.6537 |
| asc+desc, random_clouds, DA | 0.6030 | 0.6742 | 0.6824 | 0.7641 | 0.8065 | 0.8238 | 0.7683 | 0.8239 | 0.7958 | 0.7724 |
| asc+desc, random_clouds | 0.6465 | 0.7092 | 0.7203 | 0.7751 | 0.8118 | 0.8299 | 0.8036 | 0.8282 | 0.8044 | 0.7825 |
| asc+desc, random_fully_masked, DA | 0.5688 | 0.6508 | 0.6544 | 0.7270 | 0.7681 | 0.7863 | 0.7510 | 0.7888 | 0.7631 | 0.7442 |
| mix_closest, random_fully_masked | 0.5599 | 0.6461 | 0.6525 | 0.7181 | 0.7644 | 0.7838 | 0.7468 | 0.7834 | 0.7687 | 0.7437 |
| mix_closest, random_fully_masked, DA | 0.5957 | 0.6732 | 0.6804 | 0.7427 | 0.7731 | 0.7946 | 0.7687 | 0.7915 | 0.7780 | 0.7596 |
| coherence_only | 0.5442 | 0.6173 | 0.6230 | 0.7065 | 0.7719 | 0.7916 | 0.7376 | 0.7951 | 0.7420 | 0.7191 |

#### Masquage : Consecutive Fully Masked

| Modèle | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 0.4890 | 0.5622 | 0.5759 | 0.6313 | 0.6777 | 0.6957 | 0.6554 | 0.6931 | 0.6560 | 0.6389 |
| asc+desc, random_clouds, DA | 0.5554 | 0.6186 | 0.6281 | 0.6980 | 0.7363 | 0.7528 | 0.7033 | 0.7521 | 0.7281 | 0.7087 |
| asc+desc, random_clouds | 0.5994 | 0.6545 | 0.6665 | 0.7108 | 0.7423 | 0.7595 | 0.7379 | 0.7569 | 0.7384 | 0.7200 |
| asc+desc, random_fully_masked, DA | 0.5407 | 0.6169 | 0.6228 | 0.6859 | 0.7279 | 0.7461 | 0.7141 | 0.7472 | 0.7246 | 0.7061 |
| mix_closest, random_fully_masked | 0.5298 | 0.6106 | 0.6189 | 0.6737 | 0.7221 | 0.7415 | 0.7086 | 0.7403 | 0.7242 | 0.7021 |
| mix_closest, random_fully_masked, DA | 0.5520 | 0.6249 | 0.6344 | 0.6876 | 0.7230 | 0.7455 | 0.7217 | 0.7414 | 0.7230 | 0.7078 |
| coherence_only | 0.5041 | 0.5698 | 0.5769 | 0.6494 | 0.7073 | 0.7260 | 0.6789 | 0.7282 | 0.6827 | 0.6643 |

### PSNR par bande

#### Masquage : Random Fully Masked

| Modèle | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 17.89 | 18.83 | 18.19 | 18.51 | 18.88 | 19.57 | 18.18 | 19.40 | 19.55 | 18.82 |
| asc+desc, random_clouds, DA | 34.46 | 34.39 | 32.76 | 32.99 | 30.22 | 29.85 | 27.67 | 29.22 | 30.95 | 31.12 |
| asc+desc, random_clouds | 35.51 | 34.84 | 33.39 | 33.45 | 30.30 | 29.91 | 28.00 | 29.27 | 31.08 | 31.46 |
| asc+desc, random_fully_masked, DA | 34.53 | 35.53 | 33.33 | 33.43 | 30.97 | 30.64 | 28.76 | 30.10 | 32.23 | 32.08 |
| mix_closest, random_fully_masked | 34.17 | 35.18 | 32.95 | 33.01 | 30.67 | 30.29 | 28.58 | 29.82 | 31.83 | 31.84 |
| mix_closest, random_fully_masked, DA | 34.67 | 35.39 | 33.37 | 34.09 | 30.72 | 30.25 | 28.56 | 29.68 | 31.72 | 32.40 |
| coherence_only | 33.87 | 34.31 | 32.47 | 32.24 | 30.00 | 29.69 | 27.70 | 29.14 | 30.42 | 30.79 |

#### Masquage : Consecutive Fully Masked

| Modèle | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 18.38 | 19.30 | 18.68 | 18.98 | 19.30 | 19.97 | 18.60 | 19.80 | 19.97 | 19.26 |
| asc+desc, random_clouds, DA | 29.70 | 29.59 | 28.78 | 28.41 | 26.04 | 25.89 | 23.90 | 25.19 | 26.85 | 27.59 |
| asc+desc, random_clouds | 30.62 | 30.08 | 29.43 | 28.97 | 26.28 | 26.11 | 24.35 | 25.41 | 27.10 | 27.96 |
| asc+desc, random_fully_masked, DA | 32.45 | 33.60 | 32.13 | 32.01 | 30.26 | 30.19 | 28.35 | 29.60 | 31.70 | 31.76 |
| mix_closest, random_fully_masked | 31.80 | 32.84 | 31.48 | 31.17 | 29.47 | 29.43 | 27.84 | 28.99 | 30.99 | 31.23 |
| mix_closest, random_fully_masked, DA | 31.45 | 32.21 | 30.95 | 31.39 | 29.19 | 29.03 | 27.44 | 28.44 | 29.57 | 30.87 |
| coherence_only | 29.13 | 29.34 | 28.41 | 27.64 | 25.67 | 25.59 | 23.80 | 24.99 | 26.29 | 27.22 |

### MAE par bande

#### Masquage : Random Fully Masked

| Modèle | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 1856.2 | 1805.0 | 1816.4 | 1763.1 | 1637.1 | 1580.3 | 1598.4 | 1558.8 | 1533.4 | 1545.4 |
| asc+desc, random_clouds, DA | 159.2 | 165.7 | 184.5 | 200.6 | 283.0 | 304.8 | 352.0 | 326.4 | 261.0 | 225.7 |
| asc+desc, random_clouds | 142.9 | 160.0 | 174.2 | 193.5 | 280.2 | 301.5 | 341.7 | 323.5 | 259.7 | 219.1 |
| asc+desc, random_fully_masked, DA | 228.3 | 227.0 | 251.6 | 281.5 | 386.4 | 413.8 | 461.8 | 441.1 | 340.0 | 297.0 |
| mix_closest, random_fully_masked | 265.6 | 266.0 | 292.8 | 327.7 | 432.3 | 461.4 | 503.6 | 482.5 | 384.3 | 335.6 |
| mix_closest, random_fully_masked, DA | 283.4 | 287.3 | 315.3 | 336.5 | 425.3 | 459.0 | 516.9 | 495.4 | 427.8 | 347.8 |
| coherence_only | 165.6 | 161.0 | 184.1 | 204.9 | 278.1 | 296.9 | 342.3 | 317.5 | 262.9 | 225.7 |

#### Masquage : Consecutive Fully Masked

| Modèle | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 1536.0 | 1495.5 | 1506.4 | 1471.7 | 1389.0 | 1346.8 | 1373.2 | 1334.8 | 1303.7 | 1304.0 |
| asc+desc, random_clouds, DA | 242.5 | 257.6 | 269.6 | 306.9 | 419.6 | 449.9 | 502.3 | 481.8 | 389.9 | 321.5 |
| asc+desc, random_clouds | 224.1 | 249.8 | 257.0 | 297.0 | 415.3 | 444.6 | 488.7 | 475.6 | 384.1 | 313.9 |
| asc+desc, random_fully_masked, DA | 243.4 | 239.8 | 260.8 | 293.1 | 392.3 | 414.7 | 463.1 | 443.0 | 344.6 | 299.5 |
| mix_closest, random_fully_masked | 285.6 | 285.4 | 307.4 | 347.0 | 448.4 | 471.3 | 512.8 | 491.5 | 393.1 | 343.5 |
| mix_closest, random_fully_masked, DA | 321.4 | 324.7 | 351.8 | 373.8 | 451.2 | 480.1 | 539.6 | 518.3 | 474.1 | 376.3 |
| coherence_only | 255.1 | 259.8 | 275.2 | 319.7 | 427.5 | 455.3 | 504.8 | 485.5 | 399.1 | 328.0 |

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
