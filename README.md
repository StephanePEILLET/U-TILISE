# U-TILISE — Reconstruction sans nuages de séries temporelles Sentinel-2

Adaptation du modèle [U-TILISE](https://doi.org/10.1109/TGRS.2023.3333391) (Stucker et al., 2023)
pour la reconstruction de séries temporelles optiques **Sentinel-2** sur le territoire français,
avec guidage **Sentinel-1 SAR** (orbites ascendante + descendante).

Le pipeline complet permet de :
1. **Créer un fichier HDF5** à partir de données raster Sentinel-1/2 (GeoTIFF)
2. **Entraîner** le modèle U-TILISE (architecture v4, SAR asc+desc)
3. **Évaluer** les performances avec différentes stratégies de masquage et de fusion temporelle
4. **Inférer** à la tuile pour produire des séries temporelles complètes

---

## Table des matières

- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Données](#données)
  - [Format HDF5](#format-hdf5-mode-principal)
  - [Fichiers à plat](#fichiers-à-plat-mode-alternatif)
  - [Bandes spectrales](#bandes-spectrales)
- [Pipeline complet](#pipeline-complet)
  - [1. Création du HDF5](#1-création-du-hdf5)
  - [2. Entraînement](#2-entraînement)
  - [3. Évaluation](#3-évaluation)
  - [4. Inférence à la tuile](#4-inférence-à-la-tuile)
- [Modes de masquage](#modes-de-masquage)
- [Modes de fusion temporelle (blend)](#modes-de-fusion-temporelle-blend)
- [Notebooks de démonstration](#notebooks-de-démonstration)
- [Utilisation sur cluster (SLURM)](#utilisation-sur-cluster-slurm)
- [Références](#références)

---

## Installation

### Environnement conda

```bash
conda env create -f envs/cloud_reconstruction.yml
conda activate cloud_reconstruction
```

L'environnement nécessite Python 3.10+, PyTorch 2.x et CUDA 11.8+.

---

## Structure du projet

```
U-TILISE/
├── run_train.py                  # Point d'entrée : entraînement
├── run_eval.py                   # Point d'entrée : évaluation (métriques)
├── infer_from_tiles.py           # Point d'entrée : inférence sur tuiles complètes
│
├── configs/                      # Configurations YAML
│   ├── default.yaml              #   Paramètres par défaut du modèle
│   ├── config_run_train.yaml     #   Config de base pour l'entraînement
│   ├── config_run_eval.yaml      #   Config de base pour l'évaluation
│   ├── config_run_infer_*.yaml   #   Configs inférence tuile
│   └── jzellou/                  #   Configs spécifiques au cluster
│       ├── configs/              #     Configs YAML (train, eval, inférence)
│       └── slurms/               #     Scripts SLURM de soumission
│
├── lib/                          # Bibliothèque principale U-TILISE
│   ├── models/                   #   Architecture du modèle
│   │   ├── utilise.py            #     Encodeur spatial + LTAE + décodeur
│   │   ├── ltae_transformer.py   #     Lightweight Temporal Attention Encoder
│   │   ├── positional_encoding.py#     Encodage positionnel temporel
│   │   ├── interpolator.py       #     Interpolation triviale (baseline)
│   │   ├── make_layers.py        #     Constructeurs de couches
│   │   ├── weight_init.py        #     Initialisation des poids
│   │   └── parameters.py         #     Enums (activation, normalisation)
│   ├── metrics/                  #   Métriques de reconstruction
│   │   ├── cloud_removal.py      #     Métriques par sample (MAE, RMSE, PSNR, SSIM, SAM)
│   │   └── aggregation.py        #     Agrégation dataset (torchmetrics)
│   ├── datasets/                 #   Registre des datasets
│   │   ├── dataset_tools.py      #     Détection de frames nuageuses
│   │   └── mask_generation.py    #     Masques synthétiques
│   ├── trainer.py                #   Boucle d'entraînement (train/val)
│   ├── loss.py                   #   Fonctions de perte (L1, SSIM, NDVI, R²)
│   ├── eval_tools.py             #   Imputation fenêtre glissante + fusion
│   ├── metrics.py                #   Fonctions métriques utilitaires
│   ├── data_utils.py             #   Chargement datasets / dataloaders
│   ├── config_utils.py           #   Lecture/écriture configs (OmegaConf)
│   ├── utils.py                  #   Instanciation modèle, optimiseur, scheduler
│   ├── torch_transforms.py       #   Augmentations (rotation, flip, bruit)
│   ├── parcel_mask.py            #   Masques parcellaires agricoles (GPKG)
│   ├── visutils.py               #   Visualisation (galeries, colormaps)
│   ├── logger.py                 #   Logging et statistiques
│   └── formatter.py              #   Formatage des logs
│
├── dataloader/                   # Chargement des données
│   ├── datasets/
│   │   ├── hdf5_creator.py       #   Création HDF5 (FileScanner, HDF5Maker, merge)
│   │   ├── hdf5_reader.py        #   Lecture HDF5 (HDF5Dataset)
│   │   ├── adapter.py            #   Adaptation → format U-TILISE (SatelliteDataset)
│   │   ├── dataset_from_files.py #   Chargement direct depuis TIF
│   │   └── constants.py          #   Splits géographiques train/val/test
│   └── tools/
│       ├── data_processor.py     #   Lecture rasters (rasterio)
│       ├── mask_generation.py    #   Utilitaires de masquage
│       ├── sampling.py           #   Échantillonnage temporel
│       ├── torch_transforms.py   #   Transformations PyTorch
│       ├── type_converter.py     #   Conversion float ↔ int16
│       ├── writer.py             #   Écriture de prédictions (GeoTIFF)
│       └── positional_encoding.py#   Encodage positionnel
│
├── data/                         # Métadonnées des patches
├── envs/                         # Environnement conda
│   └── cloud_reconstruction.yml  #   Définition de l'environnement
├── notebooks/                    # Notebooks de démonstration
│   ├── training_demo.ipynb       #   Démo entraînement (subset + TensorBoard)
│   └── inference_demo.ipynb      #   Démo inférence (visualisation + métriques)
└── docs/                         # Documentation
```

---

## Données

### Format HDF5 (mode principal)

Le fichier HDF5 (`CIRCA_CR_merged.hdf5`) organise les données par hiérarchie géographique :

```
CIRCA_CR_merged.hdf5
└── MGRS_ID/                          # Zone UTM 10 km (ex: "31UDP_row-3_col-2")
    └── MGRSC_ID/                     # Sous-tuile 100 m (ex: "31UDP0307")
        └── window_x_y_w_h/           # Patch 256×256 pixels
            ├── S2/
            │   ├── S2              [T, 10, 256, 256]  int16   # 10 bandes S2
            │   ├── S2_dates        [T]                bytes   # Dates YYYYMMDD
            │   ├── cloud_mask      [T, 256, 256]      uint8   # Masque binaire
            │   └── cloud_prob      [T, 256, 256]      float32 # Proba nuageuse
            ├── S1/
            │   ├── S1_asc          [T_a, 4, 256, 256] int16   # SAR ascendant
            │   ├── S1_desc         [T_d, 4, 256, 256] int16   # SAR descendant
            │   ├── S1_dates_asc    [T_a]              bytes   # Dates ASC
            │   ├── S1_dates_desc   [T_d]              bytes   # Dates DESC
            │   └── S2_S1_pairing   JSON                       # Appariement S2↔S1
            ├── valid_obs           [T] ou [N]         int     # Observation valides
            ├── idx_good_frames     [N]                int     # Frames claires
            └── idx_cloudy_frames   [N]                int     # Frames nuageuses
```

**Différence train/val vs test** :
- **Train/Val** : seules les dates cloud-free sont stockées dans S2 → fichier plus compact.
  `valid_obs` est un vecteur binaire `[1, 1, ..., 1]` de taille N_valid.
- **Test** : toutes les dates sont stockées → nécessaire pour l'inférence en mode produit.
  `valid_obs` contient les indices des dates claires dans la séquence complète.

### Fichiers à plat (mode alternatif)

Le module `dataset_from_files.py` permet de charger les données directement depuis des
répertoires de GeoTIFF, sans passer par le HDF5. Utile pour le prototypage ou l'inférence
sur de nouvelles zones.

### Bandes spectrales

| Capteur | Bandes | Nombre |
|---------|--------|--------|
| **Sentinel-2** | B02, B03, B04, B05, B06, B07, B08, B08A, B11, B12 | 10 |
| **Sentinel-1** (par orbite) | VV, VH, Cohérence VV, Cohérence VH | 4 |

En mode `asc+desc`, les 8 bandes SAR (4 ASC + 4 DESC) sont concaténées aux 10 bandes S2,
soit **18 canaux d'entrée** au total. Le modèle produit **10 canaux de sortie** (bandes S2
reconstruites).

---

## Pipeline complet

### 1. Création du HDF5

Le module `dataloader/datasets/hdf5_creator.py` consolide toute la chaîne de
préparation des données :

```python
from dataloader.datasets.hdf5_creator import HDF5Maker, merge_hdf5_files

# Étape 1 : Créer les HDF5 individuels par zone MGRS
maker = HDF5Maker(
    data_optique="chemin/vers/optique_dataset/",
    data_radar="chemin/vers/radar_dataset_v4/",
    image_size=[256, 256],
    hdf5_folder="chemin/vers/sortie_hdf5/",
    use_sar=True,
    channels="all",
    filter_settings={"type": "cloud-free", "min_length": 5},
)
maker.load_items_to_hdf5()

# Étape 2 : Fusionner les HDF5 individuels en un seul fichier
merge_hdf5_files(
    hdf5_folder="chemin/vers/sortie_hdf5/",
    output_file="CIRCA_CR_merged.hdf5"
)
```

Les étapes internes du pipeline :
1. **Scan** des répertoires TIF (Sentinel-2 + Sentinel-1)
2. **Appariement** S2↔S1 : pour chaque date S2, sélection de la date S1 la plus proche (ASC et DESC séparément)
3. **Filtrage** des frames nuageuses (basé sur le masque nuageux)
4. **Écriture** en HDF5 par patch 256×256
5. **Fusion** de tous les HDF5 individuels

### 2. Entraînement

```bash
python run_train.py configs/jzellou/configs/train_v4_asc_desc_random_clouds_combined.yaml \
    --save_dir /chemin/vers/sortie/
```

Le script fusionne automatiquement `configs/default.yaml` + `configs/config_run_train.yaml`
+ la config spécifique fournie en argument.

**Configuration v4 asc+desc (meilleur modèle)** :
- Architecture élargie : encodeur `[64, 96, 128, 192, 256]`, décodeur `[96, 128, 192, 192, 256]`
- 8 têtes d'attention, 8 groupes de normalisation
- SAR asc+desc (18 canaux entrée, 10 canaux sortie)
- Loss combinée : L1 + SSIM + L1 sur pixels masqués
- Scheduler cyclique : CosineAnnealingWarmRestarts ($T_0 = 40$, $T_{mult} = 2$)
- 200 epochs, batch_size=4, gradient accumulation=2
- Séquences de 10 frames à l'entraînement

À la fin de l'entraînement, une évaluation automatique est lancée sur les 3 types de masquage.

### 3. Évaluation

```bash
python run_eval.py configs/jzellou/configs/config_run_eval_v4_asc_desc_rfm.yaml utilise
```

Paramètres de la config d'évaluation :
- `checkpoint` : chemin vers le modèle entraîné (.pth)
- `config_file` : config sauvegardée durant l'entraînement (pour l'architecture)
- `mask_type` : type de masquage (voir section dédiée)
- `max_seq_length` : fenêtre temporelle (14 par défaut en évaluation)
- `blend_mode` : stratégie de fusion (voir section dédiée)
- `save_dir` : répertoire de sortie des métriques

**Métriques calculées** : MAE, MSE, RMSE, PSNR, SSIM, SAM, R².

### 4. Inférence à la tuile

```bash
python infer_from_tiles.py configs/jzellou/configs/config_run_infer_from_tiles_v4_asc_desc.yaml utilise
```

Produit des GeoTIFF de reconstruction pour chaque tuile MGRSC. Le script :
- Découpe chaque tuile en patches de 256×256 avec chevauchement
- Applique le modèle par fenêtre glissante
- Fusionne les patches reconstruits en une tuile complète
- Écrit les résultats en GeoTIFF géoréférencé

---

## Modes de masquage

| Mode | Abréviation | Description | Usage |
|------|-------------|-------------|-------|
| `random_clouds` | rc | Masques nuageux aléatoires superposés | Entraînement |
| `random_fully_masked` | rfm | Dates entières masquées aléatoirement | Évaluation |
| `consecutive_fully_masked` | cfm | Bloc de dates consécutives masquées | Évaluation |
| `real_clouds` | — | Masques nuageux réels du HDF5 | Inférence produit |

---

## Modes de fusion temporelle (blend)

Lors de l'inférence, la séquence complète est traitée par **fenêtre glissante** de taille
`max_seq_length` (14 frames en évaluation). Le `blend_mode` détermine comment les
prédictions des fenêtres qui se chevauchent sont combinées :

| Mode | Description |
|------|-------------|
| `switch` | Transition dure au frame minimisant l'erreur entre fenêtres adjacentes |
| `center` | Pondération triangulaire favorisant le centre de chaque fenêtre |
| `center_only` | Ne conserve que les N frames centrales par fenêtre |
| `iterative` | Multi-passes : les prédictions sont réinjectées comme observations pour les passes suivantes. Le nombre de passes est calculé dynamiquement selon la plus longue lacune consécutive |

---

## Notebooks de démonstration

Deux notebooks Jupyter sont disponibles dans `notebooks/` :

| Notebook | Description |
|----------|-------------|
| `training_demo.ipynb` | Entraînement interactif sur un sous-ensemble réduit avec suivi TensorBoard |
| `inference_demo.ipynb` | Chargement d'un checkpoint, inférence, visualisation des résultats et des masques d'attention, calcul des métriques |

---

## Utilisation sur cluster (SLURM)

Les scripts SLURM sont dans `configs/jzellou/slurms/`. Exemples :

```bash
# Entraînement
sbatch configs/jzellou/slurms/train_v4_asc_desc_random_clouds_combined.slurm

# Évaluation (rfm = random_fully_masked, cfm = consecutive_fully_masked)
sbatch configs/jzellou/slurms/metrics_v4_asc_desc_rfm.slurm
sbatch configs/jzellou/slurms/metrics_v4_asc_desc_cfm.slurm

# Évaluation avec mode de fusion alternatif
sbatch configs/jzellou/slurms/metrics_v4_asc_desc_rfm_blend.slurm
sbatch configs/jzellou/slurms/metrics_v4_asc_desc_cfm_iterative.slurm

# Inférence à la tuile
sbatch configs/jzellou/slurms/infer_from_tiles_v4_asc_desc.slurm
```

---

## Références

Ce code est une adaptation de U-TILISE pour le projet CIRCA (3STR — RPG).

```bibtex
@article{stucker2023u,
  title={{U-TILISE}: A Sequence-to-sequence Model for Cloud Removal
         in Optical Satellite Time Series},
  author={Stucker, Corinne and Garnot, Vivien Sainte Fare and Schindler, Konrad},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023},
  volume={61}
}
```

U-TILISE étend l'architecture [U-TAE](https://github.com/VSainteuf/utae-paps) en un modèle
séquence-à-séquence 3D spatio-temporel complet.
