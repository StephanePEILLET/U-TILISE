# Rapport de Métriques — Cloud Reconstruction U-TILISE

**Source des résultats :** logs SLURM dans `/mnt/stores/store_dai/tmp/speillet/logs` + JSON dans `/mnt/DATA_10T/data_rpg/outputs/U-TILISE/metrics`

## 1. Récapitulatif Global

### Masquage : Random Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 114.7 | 207.7 | 35.27 | 0.7129 | 0.0385 | 0.9535 |
| asc+desc, random_fully_masked, DA | 126.9 | 250.0 | 34.54 | 0.7202 | 0.0404 | 0.9358 |
| coherence_only | 270.2 | 511.9 | 30.89 | 0.7045 | 0.0698 | 0.7948 |
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
| **v4** asc+desc, combined, blend | 109.4 | 193.4 | 35.60 | 0.7386 | 0.0352 | 0.9570 |
| **v4** asc+desc, combined, NDVI+R² | 113.0 | 210.7 | 35.25 | 0.7278 | 0.0377 | 0.9503 |
| **v4** mix_closest, combined, NDVI+R² | 109.3 | 195.7 | 35.52 | 0.7401 | 0.0348 | 0.9582 |

### Masquage : Consecutive Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 131.9 | 278.3 | 33.42 | 0.6548 | 0.0411 | 0.9187 |
| asc+desc, random_fully_masked, DA | 145.6 | 321.6 | 33.47 | 0.6832 | 0.0395 | 0.9077 |
| coherence_only | 389.1 | 837.3 | 26.54 | 0.6486 | 0.1003 | 0.6211 |
| **v2** mix_closest, random_fully_masked | 151.2 | 329.0 | 33.10 | 0.6783 | 0.0412 | 0.9038 |
| **v2** mix_closest, random_clouds | 117.0 | 262.6 | 34.26 | 0.6824 | 0.0360 | 0.9236 |
| **v2** asc+desc, random_fully_masked | 152.8 | 336.6 | 33.02 | 0.6760 | 0.0403 | 0.9000 |
| **v2** asc+desc, random_clouds | 125.6 | 270.7 | 33.75 | 0.6586 | 0.0393 | 0.9204 |
| **v3** mix_closest, random_clouds | 135.8 | 286.3 | 33.27 | 0.6651 | 0.0419 | 0.9132 |
| **v3** loss (L1+SSIM+L1_occ) | 125.2 | 269.3 | 33.87 | 0.6987 | 0.0377 | 0.9219 |
| **v3** wider | 130.6 | 287.7 | 33.45 | 0.6702 | 0.0376 | 0.9139 |
| **v3** combined (wider+cyclic+loss) | 122.9 | 266.0 | 33.82 | 0.6901 | 0.0377 | 0.9241 |
| **v3** cyclic | 119.1 | 273.0 | 34.09 | 0.6826 | 0.0354 | 0.9201 |
| **v4** asc+desc, combined | 127.3 | 266.6 | 33.54 | 0.6822 | 0.0381 | 0.9245 |
| **v4** mix_closest, combined | 128.8 | 271.2 | 33.57 | 0.6832 | 0.0388 | 0.9224 |
| **v4** asc+desc, combined, blend | 126.2 | 264.7 | 33.62 | 0.6837 | 0.0380 | 0.9255 |
| **v4** asc+desc, combined, NDVI+R² | 123.8 | 271.6 | 33.81 | 0.6801 | 0.0381 | 0.9207 |
| **v4** mix_closest, combined, NDVI+R² | 124.4 | 266.9 | 33.62 | 0.6884 | 0.0371 | 0.9263 |

## 2. Métriques sur pixels reconstruits (Occluded)

### Masquage : Random Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 352.7 | 520.1 | 26.70 | 0.4409 | 0.0886 | 0.7902 |
| asc+desc, random_fully_masked, DA | 435.8 | 658.9 | 25.26 | 0.4019 | 0.1014 | 0.7359 |
| coherence_only | 1083.9 | 1389.1 | 19.50 | 0.3465 | 0.2370 | 0.4459 |
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
| **v4** asc+desc, combined, blend | 275.9 | 438.2 | 28.55 | 0.5388 | 0.0636 | 0.8389 |
| **v4** asc+desc, combined, NDVI+R² | 353.0 | 534.2 | 26.52 | 0.4700 | 0.0882 | 0.7811 |
| **v4** mix_closest, combined, NDVI+R² | 289.8 | 452.2 | 28.20 | 0.5259 | 0.0649 | 0.8332 |

### Masquage : Consecutive Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 594.2 | 880.5 | 22.20 | 0.2190 | 0.1284 | 0.5547 |
| asc+desc, random_fully_masked, DA | 698.2 | 1026.0 | 22.28 | 0.2904 | 0.1122 | 0.5836 |
| coherence_only | 2666.6 | 2956.3 | 10.97 | 0.1038 | 0.5976 | 0.1410 |
| **v2** mix_closest, random_fully_masked | 704.1 | 1036.2 | 22.24 | 0.3089 | 0.1106 | 0.5790 |
| **v2** mix_closest, random_clouds | 542.1 | 847.0 | 22.71 | 0.2766 | 0.1151 | 0.5809 |
| **v2** asc+desc, random_fully_masked | 729.6 | 1070.8 | 21.91 | 0.3149 | 0.1111 | 0.5681 |
| **v2** asc+desc, random_clouds | 567.2 | 858.5 | 22.49 | 0.2374 | 0.1191 | 0.5722 |
| **v3** mix_closest, random_clouds | 626.4 | 907.3 | 21.91 | 0.2504 | 0.1291 | 0.5601 |
| **v3** loss (L1+SSIM+L1_occ) | 568.2 | 855.1 | 22.60 | 0.3146 | 0.1093 | 0.5880 |
| **v3** wider | 619.6 | 926.9 | 21.83 | 0.2896 | 0.1141 | 0.5720 |
| **v3** combined (wider+cyclic+loss) | 521.7 | 829.7 | 22.99 | 0.3176 | 0.1063 | 0.5993 |
| **v3** cyclic | 583.8 | 894.3 | 22.18 | 0.2528 | 0.1187 | 0.5675 |
| **v4** asc+desc, combined | 492.8 | 800.8 | 23.48 | 0.3306 | 0.0956 | 0.6241 |
| **v4** mix_closest, combined | 550.3 | 842.7 | 22.77 | 0.3133 | 0.1075 | 0.5990 |
| **v4** asc+desc, combined, blend | 486.8 | 794.2 | 23.59 | 0.3349 | 0.0954 | 0.6289 |
| **v4** asc+desc, combined, NDVI+R² | 539.1 | 854.5 | 22.81 | 0.3031 | 0.1074 | 0.5924 |
| **v4** mix_closest, combined, NDVI+R² | 494.9 | 808.7 | 23.40 | 0.3335 | 0.0946 | 0.6235 |

## 3. Occluded vs Observed (détail)

### Masquage : Random Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | 352.7 | 520.1 | 26.70 | 0.4409 | 0.0886 | 0.7902 |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | 69.2 | 99.0 | 40.19 | 0.8106 | 0.0288 | 0.9911 |
| asc+desc, random_fully_masked, DA | Occluded | 435.8 | 658.9 | 25.26 | 0.4019 | 0.1014 | 0.7359 |
| asc+desc, random_fully_masked, DA | Observed | 67.7 | 98.3 | 40.33 | 0.8276 | 0.0290 | 0.9891 |
| coherence_only | Occluded | 1083.9 | 1389.1 | 19.50 | 0.3465 | 0.2370 | 0.4459 |
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
| **v4** asc+desc, combined, blend | Occluded | 275.9 | 438.2 | 28.55 | 0.5388 | 0.0636 | 0.8389 |
| **v4** asc+desc, combined, blend | Observed | 77.6 | 110.9 | 39.29 | 0.8182 | 0.0296 | 0.9860 |
| **v4** asc+desc, combined, NDVI+R² | Occluded | 353.0 | 534.2 | 26.52 | 0.4700 | 0.0882 | 0.7811 |
| **v4** asc+desc, combined, NDVI+R² | Observed | 68.0 | 98.0 | 40.37 | 0.8203 | 0.0282 | 0.9885 |
| **v4** mix_closest, combined, NDVI+R² | Occluded | 289.8 | 452.2 | 28.20 | 0.5259 | 0.0649 | 0.8332 |
| **v4** mix_closest, combined, NDVI+R² | Observed | 75.0 | 109.3 | 39.40 | 0.8226 | 0.0289 | 0.9882 |

### Masquage : Consecutive Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | 594.2 | 880.5 | 22.20 | 0.2190 | 0.1284 | 0.5547 |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | 69.4 | 99.4 | 40.16 | 0.8098 | 0.0289 | 0.9910 |
| asc+desc, random_fully_masked, DA | Occluded | 698.2 | 1026.0 | 22.28 | 0.2904 | 0.1122 | 0.5836 |
| asc+desc, random_fully_masked, DA | Observed | 67.9 | 98.5 | 40.31 | 0.8272 | 0.0290 | 0.9890 |
| coherence_only | Occluded | 2666.6 | 2956.3 | 10.97 | 0.1038 | 0.5976 | 0.1410 |
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
| **v4** asc+desc, combined | Occluded | 492.8 | 800.8 | 23.48 | 0.3306 | 0.0956 | 0.6241 |
| **v4** asc+desc, combined | Observed | 78.2 | 111.8 | 39.22 | 0.8168 | 0.0297 | 0.9857 |
| **v4** mix_closest, combined | Occluded | 550.3 | 842.7 | 22.77 | 0.3133 | 0.1075 | 0.5990 |
| **v4** mix_closest, combined | Observed | 72.0 | 102.9 | 39.90 | 0.8218 | 0.0291 | 0.9879 |
| **v4** asc+desc, combined, blend | Occluded | 486.8 | 794.2 | 23.59 | 0.3349 | 0.0954 | 0.6289 |
| **v4** asc+desc, combined, blend | Observed | 77.7 | 111.1 | 39.28 | 0.8176 | 0.0295 | 0.9859 |
| **v4** asc+desc, combined, NDVI+R² | Occluded | 539.1 | 854.5 | 22.81 | 0.3031 | 0.1074 | 0.5924 |
| **v4** asc+desc, combined, NDVI+R² | Observed | 68.2 | 98.4 | 40.34 | 0.8199 | 0.0282 | 0.9885 |
| **v4** mix_closest, combined, NDVI+R² | Occluded | 494.9 | 808.7 | 23.40 | 0.3335 | 0.0946 | 0.6235 |
| **v4** mix_closest, combined, NDVI+R² | Observed | 74.5 | 108.2 | 39.46 | 0.8230 | 0.0288 | 0.9884 |

## 4. Meilleurs modèles — Métriques Occluded

### Random Fully Masked

#### Classement (pixels occluded, score = rang moyen pondéré)

| # | Modèle | Score | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **v4** asc+desc, combined, blend | 1.00 | 275.9 | 438.2 | 28.55 | 0.5388 | 0.0636 | 0.8389 |
| 2 | **v4** asc+desc, combined | 2.00 | 277.0 | 439.9 | 28.52 | 0.5383 | 0.0636 | 0.8378 |
| 3 | **v4** mix_closest, combined, NDVI+R² | 3.00 | 289.8 | 452.2 | 28.20 | 0.5259 | 0.0649 | 0.8332 |
| 4 | **v3** loss (L1+SSIM+L1_occ) | 4.56 | 351.0 | 528.1 | 26.60 | 0.4860 | 0.0861 | 0.7850 |
| 5 | **ALL_SAR_120_epochs** (mix_closest) | 5.44 | 352.7 | 520.1 | 26.70 | 0.4409 | 0.0886 | 0.7902 |

#### Détail par bande du meilleur modèle : **v4** asc+desc, combined, blend

| Métrique | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MAE | 179.7 | 188.2 | 232.6 | 238.6 | 308.4 | 345.4 | 371.0 | 347.7 | 290.4 | 257.1 |
| RMSE | 299.4 | 310.4 | 375.2 | 372.6 | 455.0 | 503.7 | 534.6 | 504.6 | 410.1 | 374.7 |
| PSNR | 34.17 | 33.30 | 30.97 | 30.91 | 28.23 | 27.19 | 26.52 | 27.09 | 28.54 | 29.45 |
| SSIM | 0.4509 | 0.4922 | 0.5021 | 0.5459 | 0.5646 | 0.5747 | 0.5558 | 0.5788 | 0.5666 | 0.5568 |
| R2 | 0.5833 | 0.6280 | 0.6639 | 0.6366 | 0.6500 | 0.6849 | 0.6787 | 0.6936 | 0.6809 | 0.7072 |

### Consecutive Fully Masked

#### Classement (pixels occluded, score = rang moyen pondéré)

| # | Modèle | Score | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **v4** asc+desc, combined, blend | 1.11 | 486.8 | 794.2 | 23.59 | 0.3349 | 0.0954 | 0.6289 |
| 2 | **v4** asc+desc, combined | 2.22 | 492.8 | 800.8 | 23.48 | 0.3306 | 0.0956 | 0.6241 |
| 3 | **v4** mix_closest, combined, NDVI+R² | 2.67 | 494.9 | 808.7 | 23.40 | 0.3335 | 0.0946 | 0.6235 |
| 4 | **v3** combined (wider+cyclic+loss) | 4.00 | 521.7 | 829.7 | 22.99 | 0.3176 | 0.1063 | 0.5993 |
| 5 | **v4** mix_closest, combined | 5.89 | 550.3 | 842.7 | 22.77 | 0.3133 | 0.1075 | 0.5990 |

#### Détail par bande du meilleur modèle : **v4** asc+desc, combined, blend

| Métrique | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MAE | 372.8 | 376.1 | 426.8 | 438.3 | 544.4 | 604.2 | 628.9 | 614.8 | 462.5 | 399.4 |
| RMSE | 726.8 | 702.4 | 750.1 | 749.9 | 813.1 | 870.8 | 898.2 | 878.2 | 637.3 | 568.2 |
| PSNR | 26.50 | 26.37 | 25.23 | 25.04 | 23.39 | 22.54 | 22.16 | 22.35 | 24.53 | 25.63 |
| SSIM | 0.2642 | 0.2945 | 0.3023 | 0.3365 | 0.3580 | 0.3650 | 0.3505 | 0.3696 | 0.3595 | 0.3488 |
| R2 | 0.2480 | 0.3002 | 0.3632 | 0.3319 | 0.3274 | 0.3592 | 0.3705 | 0.3658 | 0.3959 | 0.4254 |

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
| **v4** asc+desc, combined | ✅ | ✅ |
| **v4** mix_closest, combined | ✅ | ✅ |
| **v4** asc+desc, combined, blend | ✅ | ✅ |
| **v4** asc+desc, combined, NDVI+R² | ✅ | ✅ |
| **v4** mix_closest, combined, NDVI+R² | ✅ | ✅ |

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
