# U-TILISE -- Reconstruction sans nuages de series temporelles Sentinel-2

Adaptation du modele [U-TILISE](https://doi.org/10.1109/TGRS.2023.3333391) (Stucker et al., 2023)
pour la reconstruction de series temporelles optiques **Sentinel-2** sur le territoire francais,
avec guidage **Sentinel-1 SAR** (orbites ascendante + descendante).

Le pipeline complet permet de :
1. **Creer un fichier HDF5** a partir de donnees raster Sentinel-1/2 (GeoTIFF)
2. **Entrainer** le modele U-TILISE (architecture v4, SAR asc+desc)
3. **Evaluer** les performances avec differentes strategies de masquage et de fusion temporelle
4. **Inferer** a la tuile pour produire des series temporelles completes

---

## Table des matieres

- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Configuration](#configuration)
- [Pipeline complet](#pipeline-complet)
  - [1. Creation du HDF5](#1-creation-du-hdf5)
  - [2. Entrainement](#2-entrainement)
  - [3. Evaluation](#3-evaluation)
  - [4. Inference a la tuile](#4-inference-a-la-tuile)
- [Modes de masquage](#modes-de-masquage)
- [Modes de fusion temporelle](#modes-de-fusion-temporelle)
- [Donnees](#donnees)
- [References](#references)

---

## Installation

### Environnement conda

```bash
conda env create -f envs/cloud_reconstruction.yml
conda activate cloud_reconstruction
```

Python 3.10+, PyTorch 2.x et CUDA 11.8+ requis.

### Variables d'environnement (`.env`)

Les configurations YAML utilisent l'interpolation `${oc.env:VAR_NAME}` (OmegaConf) pour les
chemins specifiques a chaque machine. Copiez le template et adaptez les valeurs :

```bash
cp .env.example .env
# Editez .env avec vos chemins locaux
```

| Variable | Description |
|---|---|
| `HDF5_FILE` | Chemin vers le fichier HDF5 des donnees |
| `LOAD_TRANSFORMS` | Fichier JSON des transformations geometriques |
| `OUTPUT_DIR` | Repertoire racine des sorties (entrainement, metriques, inference) |
| `CHECKPOINT` | Chemin vers le checkpoint du modele (evaluation / inference) |
| `TRAIN_CONFIG` | Config d'entrainement associee au checkpoint |
| `DATA_OPTIQUE` | Repertoire des GeoTIFF Sentinel-2 (inference tuile) |
| `DATA_RADAR` | Repertoire des GeoTIFF Sentinel-1 (inference tuile) |
| `DATA_MASKS` | Repertoire des masques de reconstruction (evaluation) |

Le fichier `.env` est ignore par git (`.gitignore`) et n'est **jamais versionne**.

---

## Structure du projet

```
U-TILISE/
├── run_train.py                  # Point d'entree : entrainement
├── run_eval.py                   # Point d'entree : evaluation (metriques)
├── run_inference.py              # Point d'entree : inference sur tuiles completes
│
├── configs/                      # Configurations YAML
│   ├── default.yaml              #   Parametres par defaut (reference complete)
│   ├── config_run_train.yaml     #   Surcharges pour l'entrainement
│   ├── config_run_eval.yaml      #   Surcharges pour l'evaluation
│   └── config_run_inference.yaml #   Surcharges pour l'inference tuile
│
├── src/                          # Bibliotheque principale U-TILISE
│   ├── models/                   #   Architecture du modele
│   │   ├── utilise.py            #     Encodeur spatial + LTAE + decodeur
│   │   ├── ltae_transformer.py   #     Lightweight Temporal Attention Encoder
│   │   ├── positional_encoding.py#     Encodage positionnel temporel
│   │   ├── make_layers.py        #     Constructeurs de couches
│   │   ├── weight_init.py        #     Initialisation des poids
│   │   └── parameters.py         #     Enums (activation, normalisation)
│   ├── metrics/                  #   Metriques de reconstruction
│   │   ├── cloud_removal.py      #     Metriques par sample (MAE, RMSE, PSNR, SSIM, SAM)
│   │   └── aggregation.py        #     Agregation dataset (torchmetrics)
│   ├── data/                     #   Module donnees unifie et refactore
│   │   ├── __init__.py           #     API publique : SentinelDataset, SentinelBackend
│   │   ├── interfaces.py         #     Protocols et types communs
│   │   ├── sentinel_dataset.py   #     Adaptateur principal : toute la logique metier
│   │   ├── backends/             #     Backends de chargement de donnees
│   │   │   ├── hdf5_backend.py   #       Lecture depuis fichier HDF5
│   │   │   ├── files_backend.py  #       Chargement direct depuis fichiers TIF
│   │   │   ├── hdf5_creator.py   #       Outil de creation de fichiers HDF5
│   │   │   └── constants.py      #       Constantes globales (splits, bandes)
│   │   └── processing/           #     Outils de traitement des donnees
│   │       ├── transforms.py     #       Pretraitements et normalisations
│   │       ├── masking.py        #       Generation de masques synthetiques
│   │       ├── sampling.py       #       Echantillonnage temporel
│   │       ├── positional.py     #       Encodage positionnel
│   │       └── dataset_tools.py  #       Utilitaires communs
│   ├── trainer.py                #   Boucle d'entrainement (train/val)
│   ├── loss.py                   #   Fonctions de perte (L1, SSIM, NDVI, R2)
│   ├── eval_tools.py             #   Imputation fenetre glissante + fusion
│   ├── data_utils.py             #   Wrapper pour instancier les datasets
│   ├── config_utils.py           #   Lecture/ecriture configs (OmegaConf + .env)
│   ├── utils.py                  #   Instanciation modele, optimiseur, scheduler
│   ├── visutils.py               #   Visualisation (galeries, colormaps)
│   ├── logger.py                 #   Logging et statistiques
│   └── formatter.py              #   Formatage des logs
│
├── metadata/                     # Metadonnees des patches
├── envs/                         # Environnement conda
│   └── cloud_reconstruction.yml  #   Definition de l'environnement
└── notebooks/                    # Notebooks de demonstration
    ├── training_demo.ipynb       #   Demo entrainement (subset + TensorBoard)
    └── inference_demo.ipynb      #   Demo inference (visualisation + metriques)
```

---

## Configuration

Les configurations YAML sont fusionnees dans l'ordre : `default.yaml` < `config_run_*.yaml` < config specifique.
Seuls les parametres qui different de `default.yaml` doivent etre specifies dans les configs derivees.

La reference complete est dans [`configs/default.yaml`](configs/default.yaml). Pour creer votre propre config,
copiez l'un des fichiers `config_run_*.yaml` et modifiez les parametres souhaites.

### Parametres cles

| Section | Parametres principaux |
|---|---|
| `data:` | `hdf5_file`, `data_optique`, `data_radar`, `use_sar`, `channels`, `max_seq_length`, `blend_mode` |
| `mask:` | `mask_type`, `ratio_masked_frames`, `fill_type`, `fill_value` |
| `utilise:` | `encoder_widths`, `decoder_widths`, `n_head`, `d_k`, `n_groups`, `dropout` |
| `loss:` | `l1_loss`, `ssim_loss`, `ndvi_loss`, `temporal_r2_loss` (+ poids `*_w`) |
| `optimizer:` | `name`, `learning_rate`, `weight_decay` |
| `scheduler:` | `name`, `milestones`, `gamma` |
| `training_settings:` | `batch_size`, `num_epochs`, `accum_iter` |

---

## Pipeline complet

### 1. Creation du HDF5

Le module `src/data/backends/hdf5_creator.py` consolide la preparation des donnees :

```python
from src.data.backends.hdf5_creator import HDF5Maker, merge_hdf5_files

# Etape 1 : Creer les HDF5 individuels par zone MGRS
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

# Etape 2 : Fusionner les HDF5 individuels en un seul fichier
merge_hdf5_files(
    hdf5_folder="chemin/vers/sortie_hdf5/",
    output_file="CIRCA_CR_merged.hdf5"
)
```

Etapes internes :
1. **Scan** des repertoires TIF (Sentinel-2 + Sentinel-1)
2. **Appariement** S2/S1 : pour chaque date S2, selection de la date S1 la plus proche (ASC et DESC)
3. **Filtrage** des frames nuageuses (base sur le masque nuageux)
4. **Ecriture** en HDF5 par patch 256x256
5. **Fusion** de tous les HDF5 individuels

### 2. Entrainement

#### Prerequis
- Fichier HDF5 pret (`CIRCA_CR_merged.hdf5`)
- Variables d'environnement configurees dans `.env` (au minimum `HDF5_FILE`, `OUTPUT_DIR`)

#### Lancer l'entrainement

```bash
python run_train.py configs/config_run_train.yaml --save_dir ./outputs/train/
```

Le script fusionne automatiquement `configs/default.yaml` + `configs/config_run_train.yaml`.
A la fin de l'entrainement, une evaluation automatique est lancee sur les 3 types de masquage
(`random_clouds`, `random_fully_masked`, `consecutive_fully_masked`).

#### Parametres par defaut (config_run_train.yaml)

| Parametre | Valeur par defaut |
|---|---|
| `data.use_sar` | `mix_closest` |
| `data.max_seq_length` | 5 |
| `data.channels` | `all` |
| `mask.mask_type` | `random_fully_masked` |
| `optimizer.learning_rate` | 2e-4 |
| `scheduler.name` | `MultiStepLR` |
| `training_settings.batch_size` | 2 |
| `training_settings.num_epochs` | 1 |

#### Pour une config personnalisee

Copiez `configs/config_run_train.yaml` et modifiez les parametres :

```bash
cp configs/config_run_train.yaml configs/mon_entrainement.yaml
# Editez mon_entrainement.yaml
python run_train.py configs/mon_entrainement.yaml --save_dir ./outputs/mon_run/
```

#### Sorties

L'entrainement genere dans `--save_dir` :
- `config.yaml` : configuration finale utilisee
- `model_config.yaml` : parametres de l'architecture
- `model_parameters.txt` : architecture detaillee
- `checkpoints/Model_best.pth` : meilleur modele (selon la loss de validation)
- `test_metrics_*.json` : metriques d'evaluation automatique post-entrainement

### 3. Evaluation

#### Prerequis
- Un checkpoint entraine (`Model_best.pth`)
- La config sauvegardee durant l'entrainement (`config.yaml` dans le dossier d'experience)
- Variables d'environnement : `CHECKPOINT`, `TRAIN_CONFIG`, `HDF5_FILE`, `OUTPUT_DIR`

#### Lancer l'evaluation

```bash
python run_eval.py configs/config_run_eval.yaml
```

#### Parametres par defaut (config_run_eval.yaml)

| Parametre | Valeur par defaut |
|---|---|
| `data.use_sar` | `asc+desc` |
| `data.blend_mode` | `center` |
| `data.max_seq_length` | 14 |
| `mask.mask_type` | `random_fully_masked` |
| `test_data.split` | `test` |

#### Pour evaluer avec un autre mode de masquage

Modifiez `mask.mask_type` dans votre config ou creez une config dediee :

```yaml
# configs/config_run_eval_rfm.yaml
mask:
    mask_type: random_fully_masked

data:
    blend_mode: iterative
```

```bash
python run_eval.py configs/config_run_eval_rfm.yaml
```

#### Metriques calculees

MAE, MSE, RMSE, PSNR, SSIM, SAM, R2.

#### Sorties

- `<OUTPUT_DIR>/metrics/test_stats.json` : metriques agregees
- `<OUTPUT_DIR>/metrics/config_eval.yaml` : config d'evaluation utilisee

### 4. Inference a la tuile

#### Prerequis
- Un checkpoint entraine (`Model_best.pth`)
- La config d'entrainement associee (`config.yaml`)
- Donnees Sentinel-2 (GeoTIFF) dans un repertoire
- Donnees Sentinel-1 (GeoTIFF) dans un repertoire
- Variables d'environnement : `CHECKPOINT`, `TRAIN_CONFIG`, `DATA_OPTIQUE`, `DATA_RADAR`, `OUTPUT_DIR`

#### Lancer l'inference

```bash
python run_inference.py configs/config_run_inference.yaml
```

Le script :
1. Decoupe chaque tuile MGRS-C en patches de 256x256 avec chevauchement
2. Applique le modele par fenetre glissante
3. Fusionne les patches reconstruits en une tuile complete
4. Ecrit les resultats en GeoTIFF georeference

#### Parametres par defaut (config_run_inference.yaml)

| Parametre | Valeur par defaut |
|---|---|
| `data.use_sar` | `mix_closest` |
| `data.max_seq_length` | 5 |
| `mask.mask_type` | `random_fully_masked` |
| `test_data.split` | `test` |

#### Sorties

Les GeoTIFF de reconstruction sont ecrits dans `<OUTPUT_DIR>/inference/tiles/<nom_experience>/` :
- `pred_mgrsc_<ZONE>.tif` pour chaque tuile MGRS-C traitee

#### Utilisation programmatique

Il est aussi possible de charger les donnees directement depuis des fichiers TIF sans HDF5 :

```python
from src.data.backends.files_backend import Dataset_from_files
from src.data import SentinelDataset

backend = Dataset_from_files(
    data_optique="./data/S2/",
    data_radar="./data/S1/",
    mgrsc="31TCJ"
)

dataset = SentinelDataset(backend=backend)
sample = dataset[0]
print(sample.keys())  # x, y, masks, position_days, etc.
```

---

## Modes de masquage

| Mode | Description | Usage |
|------|-------------|-------|
| `random_clouds` | Masques nuageux aleatoires superposes | Entrainement |
| `random_fully_masked` | Dates entieres masquees aleatoirement | Evaluation |
| `consecutive_fully_masked` | Bloc de dates consecutives masquees | Evaluation |
| `real_clouds` | Masques nuageux reels du HDF5 | Inference produit |

---

## Modes de fusion temporelle

Lors de l'evaluation/inference, la sequence complete est traitee par **fenetre glissante** de taille
`max_seq_length`. Le `blend_mode` determine comment les predictions des fenetres qui se chevauchent
sont combinees :

| Mode | Description |
|------|-------------|
| `switch` | Transition dure au frame minimisant l'erreur entre fenetres adjacentes |
| `center` | Pondération triangulaire favorisant le centre de chaque fenetre |
| `center_only` | Ne conserve que les N frames centrales par fenetre |
| `iterative` | Multi-passes : les predictions sont reinjectees comme observations pour les passes suivantes |

---

## Donnees

### Bandes spectrales

| Capteur | Bandes | Nombre |
|---------|--------|--------|
| **Sentinel-2** | B02, B03, B04, B05, B06, B07, B08, B08A, B11, B12 | 10 |
| **Sentinel-1** (par orbite) | VV, VH, Coherence VV, Coherence VH | 4 |

En mode `asc+desc`, les 8 bandes SAR (4 ASC + 4 DESC) sont concatenees aux 10 bandes S2,
soit **18 canaux d'entree** au total. Le modele produit **10 canaux de sortie** (bandes S2
reconstruites).

### Mode SAR (`use_sar`)

| Valeur | Description |
|--------|-------------|
| `false` | Pas de donnees SAR |
| `asc` | SAR ascendant uniquement |
| `desc` | SAR descendant uniquement |
| `mix_closest` | SAR asc+desc, date la plus proche par orbite |
| `asc+desc` | SAR asc+desc concatene |

---

## References

Ce code est une adaptation de U-TILISE pour le projet CIRCA (3STR -- RPG).

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

U-TILISE etend l'architecture [U-TAE](https://github.com/VSainteuf/utae-paps) en un modele
sequence-a-sequence 3D spatio-temporel complet.
