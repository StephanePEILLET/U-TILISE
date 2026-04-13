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
  - [Environnement conda](#environnement-conda)
  - [Variables d'environnement (.env)](#variables-denvironnement-env)
- [Structure du projet](#structure-du-projet)
- [Référence des paramètres de configuration](#référence-des-paramètres-de-configuration)
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

### Variables d'environnement (`.env`)

Les configurations YAML utilisent l'interpolation `${oc.env:VAR_NAME}` (OmegaConf) pour les
chemins spécifiques à chaque machine. Copiez le template et adaptez les valeurs :

```bash
cp .env.example .env
# Éditez .env avec vos chemins locaux
```

Variables disponibles (voir `.env.example` pour les détails) :

| Variable | Description |
|---|---|
| `HDF5_FILE` | Chemin vers le fichier HDF5 des données |
| `LOAD_TRANSFORMS` | Fichier JSON des transformations géométriques |
| `OUTPUT_DIR` | Répertoire racine des sorties (entraînement, métriques, inférence) |
| `CHECKPOINT` | Chemin vers le checkpoint du modèle (inférence) |
| `TRAIN_CONFIG` | Config d'entraînement associée au checkpoint |
| `DATA_OPTIQUE` | Répertoire des GeoTIFF Sentinel-2 (inférence tuile) |
| `DATA_RADAR` | Répertoire des GeoTIFF Sentinel-1 (inférence tuile) |

Le fichier `.env` est ignoré par git (`.gitignore`) et n'est **jamais versionné**.

---

## 🚀 Démarrage rapide en 2 minutes

### 1. Entraîner le modèle
```bash
python run_train.py configs/config_run_train.yaml --save_dir ./outputs/train/
```

### 2. Évaluer le modèle
```bash
python run_eval.py configs/config_run_eval.yaml
```

### 3. Inférer directement depuis des fichiers TIF
```python
# Charger directement depuis des fichiers brutes SANS HDF5
from src.data.backends.files_backend import Dataset_from_files
from src.data import SentinelDataset

# Initialiser le backend de fichiers
backend = Dataset_from_files(
    data_optique="./data/S2/",
    data_radar="./data/S1/",
    mgrsc="31TCJ"
)

# ✅ Utiliser EXACTEMENT LE MÊME adaptateur
dataset = SentinelDataset(backend=backend)

# Charger un échantillon
sample = dataset[0]
print(sample.keys())  # x, y, masks, position_days, etc.
```

---

## Structure du projet

```
U-TILISE/
├── run_train.py                  # Point d'entrée : entraînement
├── run_eval.py                   # Point d'entrée : évaluation (métriques)
├── run_inference.py              # Point d'entrée : inférence sur tuiles complètes
│
├── configs/                      # Configurations YAML
│   ├── default.yaml              #   Paramètres par défaut (référence complète)
│   ├── config_run_train.yaml     #   Surcharges pour l'entraînement
│   ├── config_run_eval.yaml      #   Surcharges pour l'évaluation
│   ├── config_run_inference.yaml #   Surcharges pour l'inférence tuile
│   └── jzellou/                  #   Configs spécifiques au cluster
│       ├── configs/              #     Configs YAML (train, eval, inférence)
│       └── slurms/               #     Scripts SLURM de soumission
│
 ├── src/                          # Bibliothèque principale U-TILISE
│   ├── models/                   #   Architecture du modèle
│   │   ├── utilise.py            #     Encodeur spatial + LTAE + décodeur
│   │   ├── ltae_transformer.py   #     Lightweight Temporal Attention Encoder
│   │   ├── positional_encoding.py#     Encodage positionnel temporel
│   │   ├── make_layers.py        #     Constructeurs de couches
│   │   ├── weight_init.py        #     Initialisation des poids
│   │   └── parameters.py         #     Enums (activation, normalisation)
│   ├── metrics/                  #   Métriques de reconstruction
│   │   ├── cloud_removal.py      #     Métriques par sample (MAE, RMSE, PSNR, SSIM, SAM)
│   │   └── aggregation.py        #     Agrégation dataset (torchmetrics)
│   ├── data/                     # ✅ MODULE DONNÉES UNIFIÉ ET RÉFACTORISÉ
│   │   ├── __init__.py           #     API publique : SentinelDataset, SentinelBackend
│   │   ├── interfaces.py         #     Protocols et types communs
│   │   ├── sentinel_dataset.py   #     ✅ ADAPTATEUR PRINCIPAL : toute la logique métier
│   │   ├── backends/             #     Backends de chargement de données
│   │   │   ├── hdf5_backend.py   #       Lecture depuis fichier HDF5
│   │   │   ├── files_backend.py  #       Chargement direct depuis fichiers TIF
│   │   │   ├── hdf5_creator.py   #       Outil de création de fichiers HDF5
│   │   │   └── constants.py      #       Constantes globales (splits, bandes)
│   │   └── processing/           #     Outils de traitement des données
│   │       ├── transforms.py     #       Prétraitements et normalisations
│   │       ├── masking.py        #       Génération de masques synthétiques
│   │       ├── sampling.py       #       Échantillonnage temporel
│   │       ├── positional.py     #       Encodage positionnel
│   │       └── dataset_tools.py  #       Utilitaires communs
│   ├── trainer.py                #   Boucle d'entraînement (train/val)
│   ├── loss.py                   #   Fonctions de perte (L1, SSIM, NDVI, R²)
│   ├── eval_tools.py             #   Imputation fenêtre glissante + fusion
│   ├── data_utils.py             #   Wrapper pour instancier les datasets
│   ├── config_utils.py           #   Lecture/écriture configs (OmegaConf + .env)
│   ├── utils.py                  #   Instanciation modèle, optimiseur, scheduler
│   ├── visutils.py               #   Visualisation (galeries, colormaps)
│   ├── logger.py                 #   Logging et statistiques
│   └── formatter.py              #   Formatage des logs
│
├── metadata/                     # Métadonnées des patches
├── envs/                         # Environnement conda
│   └── cloud_reconstruction.yml  #   Définition de l'environnement
├── notebooks/                    # Notebooks de démonstration
│   ├── training_demo.ipynb       #   Démo entraînement (subset + TensorBoard)
│   └── inference_demo.ipynb      #   Démo inférence (visualisation + métriques)
└── docs/                         # Documentation
```

---

## 📚 Référence API publique

### `src.data.SentinelDataset`
Adaptateur générique pour tous les backends de données.

**Constructeur principal :**
```python
dataset = SentinelDataset(
    backend: SentinelBackend,           # N'importe quel backend
    return_valid_obs_only: bool = True, # Ne retourner que les observations valides
    max_seq_length: int | None = 30,    # Tronquer les séquences trop longues
    render_occluded_above_p: float | None = None,  # Masquer les images trop nuageuses
    mask_kwargs: dict | None = None,    # Configuration du masquage
    pe_strategy: str = "day-within-sequence",  # Stratégie d'encodage positionnel
    augment: bool | None = False,       # Activer les augmentations
    process_data: bool | None = True,   # Appliquer tout le traitement
    mask_sar: bool = False,             # Masquer aussi les bandes SAR (mode legacy)
)
```

**Constructeur rétro-compatible HDF5 :**
```python
dataset = SentinelDataset.from_hdf5(
    phase="train",
    hdf5_file="data/circa.hdf5",
    use_sar="asc+desc",
    channels="all",
    # + tous les paramètres du constructeur principal
)
```

### `SentinelBackend` (Protocole)
Interface que tous les backends doivent implémenter :
```python
class SentinelBackend(Protocol):
    def __getitem__(self, item: int) -> SampleDict: ...
    def __len__(self) -> int: ...
    @property
    def c_index_rgb(self) -> Tensor: ...
    @property
    def c_index_nir(self) -> Tensor: ...
    @property
    def num_channels(self) -> int: ...
    @property
    def phase(self) -> PhaseType: ...
```

---

## Référence des paramètres de configuration

Les configurations YAML sont fusionnées dans l'ordre : `default.yaml` ← `config_run_*.yaml` ← config spécifique.
Seuls les paramètres qui diffèrent de `default.yaml` doivent être spécifiés dans les configs dérivées.

La référence complète est dans [`configs/default.yaml`](configs/default.yaml). Voici les
paramètres les plus importants :

### `data:` — Données d'entrée

| Paramètre | Type | Description |
|---|---|---|
| `dataset` | str | Nom du dataset (`circa`) |
| `hdf5_file` | str | Chemin vers le fichier HDF5 (`${oc.env:HDF5_FILE}`) |
| `channels` | str/list | Bandes spectrales : `all`, `rgb`, `bgr_nir`, ou liste d'indices |
| `use_sar` | str/bool | Mode SAR : `false`, `asc`, `desc`, `mix_closest`, `asc+desc` |
| `max_seq_length` | int/null | Longueur max de séquence temporelle. `null` = pas de troncature |
| `pe_strategy` | str | Encodage positionnel : `day-of-year`, `day-within-sequence`, `absolute`, `enumeration` |
| `render_occluded_above_p` | float | Seuil de couverture nuageuse (0.0–1.0) pour masquer une image entière |
| `load_transforms` | str | Chemin JSON des transformations géométriques (optionnel) |
| `blend_mode` | str | Fusion temporelle en évaluation : `switch`, `center`, `center_only`, `iterative` |
| `subset` | int/bool | Sous-ensemble de N patches (`false` = tout le dataset) |

### `mask:` — Masquage synthétique

| Paramètre | Type | Description |
|---|---|---|
| `mask_type` | str | `random_clouds`, `real_clouds`, `random_fully_masked`, `consecutive_fully_masked` |
| `ratio_masked_frames` | float | Part max d'images masquées par séquence (0.0–1.0) |
| `fill_type` | str | Initialisation des pixels masqués : `fill_value`, `white_noise`, `mean` |
| `fill_value` | float | Valeur de remplissage si `fill_type == fill_value` |

### `utilise:` — Architecture du modèle

| Paramètre | Type | Description |
|---|---|---|
| `encoder_widths` | list[int] | Nombre de filtres par niveau de l'encodeur spatial |
| `decoder_widths` | list[int] | Nombre de filtres par niveau du décodeur spatial |
| `n_head` | int | Nombre de têtes d'attention dans le LTAE |
| `d_k` | int | Dimension des clés/requêtes par tête d'attention |
| `n_groups` | int | Nombre de groupes pour la normalisation de groupe |
| `dropout` | float | Dropout général dans le réseau |
| `output_activation` | str | Activation de sortie : `sigmoid`, `softmax`, `null` |

### `loss:` — Fonctions de perte

| Paramètre | Type | Description |
|---|---|---|
| `l1_loss` | bool | Perte L1 sur tous les pixels |
| `ssim_loss` | bool | Perte SSIM (similarité structurelle) |
| `ndvi_loss` | bool | Perte L1 sur le NDVI pour la cohérence végétation |
| `temporal_r2_loss` | bool | Perte (1 - R²) pour la cohérence temporelle |
| `*_w` | float | Poids associé à chaque terme de perte |

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
    min_seq_length=5,
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
python run_eval.py configs/jzellou/configs/config_run_eval_v4_asc_desc_rfm.yaml
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
python run_inference.py configs/config_run_inference.yaml
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
sbatch configs/jzellou/slurms/inference_v4_asc_desc.slurm
```

---

---

## ❓ FAQ / Dépannage

### Q : Pourquoi l'architecture utilise Composition au lieu de l'héritage ?
Parce que l'héritage rend impossible d'avoir plusieurs backends. Avec la composition :
- On peut changer de backend sans toucher à aucune autre ligne de code
- On ajoute un nouveau backend en 20 lignes de code
- Toute modification de la logique métier est répercutée automatiquement sur tous les backends

### Q : Comment ajouter un nouveau type de masque ?
Ajoutez votre fonction dans `src/data/processing/masking.py`. Elle sera automatiquement disponible dans tous les backends.

### Q : Comment ajouter un nouveau backend (S3, COG, etc.) ?
1. Créez un fichier `src/data/backends/my_backend.py`
2. Implémentez l'interface `SentinelBackend`
3. C'est tout. `SentinelDataset` fonctionnera immédiatement avec votre nouveau backend.

### Q : Pourquoi il n'y a plus de dossier `dataloader/` ?
Tout le code de chargement des données a été **factorisé et unifié** dans `src/data/`. L'ancien dossier `dataloader/` contenait 70% de code dupliqué.

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
