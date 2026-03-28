# Rapport de Métriques — Cloud Reconstruction U-TILISE

**Source des résultats :** logs SLURM dans `/mnt/stores/store_dai/tmp/speillet/logs` + JSON dans `/mnt/DATA_10T/data_rpg/outputs/U-TILISE/metrics`

## 1. Récapitulatif Global

### Masquage : Random Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 1108.7 | 2112.3 | 21.53 | 0.6394 | 0.0735 | 0.4541 |
| asc+desc, random_fully_masked, DA | — | — | — | — | — | — |
| coherence_only | 268.9 | 509.7 | 30.91 | 0.7048 | 0.0696 | 0.7959 |
| **v2** mix_closest, random_fully_masked | 129.7 | 248.6 | 34.30 | 0.7234 | 0.0417 | 0.9364 |
| **v2** mix_closest, random_clouds | 107.4 | 204.7 | 35.69 | 0.7326 | 0.0353 | 0.9527 |
| **v2** asc+desc, random_fully_masked | 129.4 | 250.8 | 34.33 | 0.7200 | 0.0408 | 0.9356 |
| **v2** asc+desc, random_clouds | 116.0 | 216.0 | 35.06 | 0.7091 | 0.0392 | 0.9478 |
| **v3** mix_closest, random_clouds | 122.6 | 226.6 | 34.66 | 0.7039 | 0.0420 | 0.9432 |

### Masquage : Consecutive Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 823.9 | 1820.3 | 22.07 | 0.6275 | 0.0598 | 0.4705 |
| asc+desc, random_fully_masked, DA | 145.6 | 321.6 | 33.47 | 0.6832 | 0.0395 | 0.9077 |
| coherence_only | 389.1 | 837.2 | 26.54 | 0.6488 | 0.1002 | 0.6212 |
| **v2** mix_closest, random_fully_masked | 151.2 | 329.0 | 33.10 | 0.6783 | 0.0412 | 0.9038 |
| **v2** mix_closest, random_clouds | 117.0 | 262.6 | 34.26 | 0.6824 | 0.0360 | 0.9236 |
| **v2** asc+desc, random_fully_masked | 152.8 | 336.6 | 33.02 | 0.6760 | 0.0403 | 0.9000 |
| **v2** asc+desc, random_clouds | 125.6 | 270.7 | 33.75 | 0.6586 | 0.0393 | 0.9204 |
| **v3** mix_closest, random_clouds | 135.8 | 286.3 | 33.27 | 0.6651 | 0.0419 | 0.9132 |

## 2. Métriques sur pixels reconstruits (Occluded)

### Masquage : Random Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 6416.1 | 6750.5 | 3.63 | 0.0806 | 0.2969 | 0.0243 |
| asc+desc, random_fully_masked, DA | — | — | — | — | — | — |
| coherence_only | 1078.2 | 1382.8 | 19.53 | 0.3466 | 0.2360 | 0.4474 |
| **v2** mix_closest, random_fully_masked | 421.7 | 636.5 | 25.48 | 0.4587 | 0.0976 | 0.7420 |
| **v2** mix_closest, random_clouds | 360.5 | 538.3 | 26.36 | 0.4566 | 0.0903 | 0.7770 |
| **v2** asc+desc, random_fully_masked | 432.5 | 651.6 | 25.34 | 0.4591 | 0.0980 | 0.7394 |
| **v2** asc+desc, random_clouds | 379.2 | 556.4 | 26.01 | 0.4229 | 0.0965 | 0.7644 |
| **v3** mix_closest, random_clouds | 406.1 | 585.0 | 25.54 | 0.3765 | 0.1065 | 0.7422 |

### Masquage : Consecutive Fully Masked

| Modèle | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | 6485.9 | 6753.7 | 3.60 | 0.0820 | 0.2818 | 0.0316 |
| asc+desc, random_fully_masked, DA | 698.2 | 1026.0 | 22.28 | 0.2904 | 0.1122 | 0.5836 |
| coherence_only | 2665.8 | 2955.5 | 10.97 | 0.1041 | 0.5966 | 0.1401 |
| **v2** mix_closest, random_fully_masked | 704.1 | 1036.2 | 22.24 | 0.3089 | 0.1106 | 0.5790 |
| **v2** mix_closest, random_clouds | 542.1 | 847.0 | 22.71 | 0.2766 | 0.1151 | 0.5809 |
| **v2** asc+desc, random_fully_masked | 729.6 | 1070.8 | 21.91 | 0.3149 | 0.1111 | 0.5681 |
| **v2** asc+desc, random_clouds | 567.2 | 858.5 | 22.49 | 0.2374 | 0.1191 | 0.5722 |
| **v3** mix_closest, random_clouds | 626.4 | 907.3 | 21.91 | 0.2504 | 0.1291 | 0.5601 |

## 3. Occluded vs Observed (détail)

### Masquage : Random Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | 6416.1 | 6750.5 | 3.63 | 0.0806 | 0.2969 | 0.0243 |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | 69.9 | 100.2 | 40.08 | 0.8085 | 0.0290 | 0.9908 |
| asc+desc, random_fully_masked, DA | Occluded | — | — | — | — | — | — |
| asc+desc, random_fully_masked, DA | Observed | — | — | — | — | — | — |
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

### Masquage : Consecutive Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | 6485.9 | 6753.7 | 3.60 | 0.0820 | 0.2818 | 0.0316 |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | 70.1 | 100.6 | 40.05 | 0.8078 | 0.0291 | 0.9907 |
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

## 4. Meilleurs modèles — Métriques Occluded

### Random Fully Masked

#### Classement (pixels occluded, score = rang moyen pondéré)

| # | Modèle | Score | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **v2** mix_closest, random_clouds | 1.22 | 360.5 | 538.3 | 26.36 | 0.4566 | 0.0903 | 0.7770 |
| 2 | **v2** asc+desc, random_clouds | 2.22 | 379.2 | 556.4 | 26.01 | 0.4229 | 0.0965 | 0.7644 |
| 3 | **v3** mix_closest, random_clouds | 3.44 | 406.1 | 585.0 | 25.54 | 0.3765 | 0.1065 | 0.7422 |
| 4 | **v2** mix_closest, random_fully_masked | 3.67 | 421.7 | 636.5 | 25.48 | 0.4587 | 0.0976 | 0.7420 |
| 5 | **v2** asc+desc, random_fully_masked | 4.44 | 432.5 | 651.6 | 25.34 | 0.4591 | 0.0980 | 0.7394 |

#### Détail par bande du meilleur modèle : **v2** mix_closest, random_clouds

| Métrique | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MAE | 209.0 | 221.9 | 292.8 | 284.3 | 400.8 | 478.0 | 509.1 | 480.7 | 375.2 | 353.0 |
| RMSE | 335.6 | 349.1 | 447.9 | 424.4 | 550.2 | 644.7 | 679.5 | 645.2 | 512.2 | 485.9 |
| PSNR | 32.55 | 31.66 | 28.77 | 29.24 | 26.16 | 24.65 | 24.11 | 24.60 | 26.52 | 27.09 |
| SSIM | 0.3661 | 0.4102 | 0.4067 | 0.4740 | 0.4865 | 0.4885 | 0.4686 | 0.4997 | 0.4937 | 0.4721 |
| R2 | 0.4490 | 0.5062 | 0.4971 | 0.5012 | 0.5062 | 0.5218 | 0.5210 | 0.5419 | 0.5152 | 0.5252 |

### Consecutive Fully Masked

#### Classement (pixels occluded, score = rang moyen pondéré)

| # | Modèle | Score | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **v2** mix_closest, random_clouds | 1.89 | 542.1 | 847.0 | 22.71 | 0.2766 | 0.1151 | 0.5809 |
| 2 | asc+desc, random_fully_masked, DA | 3.00 | 698.2 | 1026.0 | 22.28 | 0.2904 | 0.1122 | 0.5836 |
| 3 | **v2** asc+desc, random_clouds | 3.22 | 567.2 | 858.5 | 22.49 | 0.2374 | 0.1191 | 0.5722 |
| 4 | **v2** mix_closest, random_fully_masked | 3.67 | 704.1 | 1036.2 | 22.24 | 0.3089 | 0.1106 | 0.5790 |
| 5 | **v3** mix_closest, random_clouds | 4.56 | 626.4 | 907.3 | 21.91 | 0.2504 | 0.1291 | 0.5601 |

#### Détail par bande du meilleur modèle : **v2** mix_closest, random_clouds

| Métrique | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B8A | B11 | B12 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MAE | 387.5 | 395.8 | 456.7 | 463.1 | 614.7 | 695.0 | 727.2 | 709.5 | 516.4 | 454.6 |
| RMSE | 741.8 | 717.1 | 771.7 | 762.6 | 874.1 | 953.4 | 987.4 | 965.6 | 691.7 | 614.9 |
| PSNR | 26.04 | 25.89 | 24.64 | 24.59 | 22.42 | 21.42 | 21.04 | 21.23 | 23.70 | 24.76 |
| SSIM | 0.2018 | 0.2304 | 0.2403 | 0.2770 | 0.2991 | 0.3060 | 0.2901 | 0.3136 | 0.3112 | 0.2961 |
| R2 | 0.2062 | 0.2555 | 0.3012 | 0.2817 | 0.2395 | 0.2634 | 0.2745 | 0.2748 | 0.3254 | 0.3512 |

## 5. Disponibilité des résultats

| Modèle | Random Fully Masked | Consecutive Fully Masked |
|:---|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | ✅ | ✅ |
| asc+desc, random_clouds, DA | ❌ | ❌ |
| asc+desc, random_clouds | ❌ | ❌ |
| asc+desc, random_fully_masked, DA | ❌ | ✅ |
| mix_closest, random_fully_masked | ❌ | ❌ |
| mix_closest, random_fully_masked, DA | ❌ | ❌ |
| coherence_only | ✅ | ✅ |
| **v2** mix_closest, random_fully_masked | ✅ | ✅ |
| **v2** mix_closest, random_clouds | ✅ | ✅ |
| **v2** asc+desc, random_fully_masked | ✅ | ✅ |
| **v2** asc+desc, random_clouds | ✅ | ✅ |
| **v3** mix_closest, random_clouds | ✅ | ✅ |
| **v3** loss (L1+SSIM+L1_occ) | ❌ | ❌ |
| **v3** wider | ❌ | ❌ |
| **v3** combined (wider+cyclic+loss) | ❌ | ❌ |
| **v3** cyclic | ❌ | ❌ |

## 6. Métriques à la Parcelle (RPG)

### Masquage : Random Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Global | 1116.8 | 2113.2 | 21.58 | 0.6394 | 0.0739 | 0.4569 |
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | 6419.1 | 6743.0 | 3.59 | 0.0806 | 0.3005 | 0.0218 |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | 70.2 | 100.3 | 40.08 | 0.8085 | 0.0287 | 0.9912 |
| **v2** mix_closest, random_clouds | Global | — | — | — | — | — | — |
| **v2** mix_closest, random_clouds | Occluded | — | — | — | — | — | — |
| **v2** mix_closest, random_clouds | Observed | — | — | — | — | — | — |

### Masquage : Consecutive Fully Masked

| Modèle | Type | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Global | — | — | — | — | — | — |
| **ALL_SAR_120_epochs** (mix_closest) | Occluded | — | — | — | — | — | — |
| **ALL_SAR_120_epochs** (mix_closest) | Observed | — | — | — | — | — | — |
| **v2** mix_closest, random_clouds | Global | — | — | — | — | — | — |
| **v2** mix_closest, random_clouds | Occluded | — | — | — | — | — | — |
| **v2** mix_closest, random_clouds | Observed | — | — | — | — | — | — |

_Aucun résultat parcelle disponible._


## 7. Impact du Filtrage des Pixels Noirs (nodata)

Comparaison des métriques **avant** et **après** exclusion des pixels où toutes les bandes = 0 dans la cible.

| Modèle | Masquage | Version | MAE | RMSE | PSNR | SSIM | SAM | R2 |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ALL_SAR_120_epochs** (mix_closest) | Random Fully Masked | Sans filtrage | — | — | — | — | — | — |
| **ALL_SAR_120_epochs** (mix_closest) | Random Fully Masked | Avec filtrage | — | — | — | — | — | — |
| **ALL_SAR_120_epochs** (mix_closest) | Consecutive Fully Masked | Sans filtrage | — | — | — | — | — | — |
| **ALL_SAR_120_epochs** (mix_closest) | Consecutive Fully Masked | Avec filtrage | — | — | — | — | — | — |

_Aucune comparaison nodata disponible._

