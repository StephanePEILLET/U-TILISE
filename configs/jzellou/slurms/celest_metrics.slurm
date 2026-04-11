#!/bin/bash
#SBATCH --job-name=celest_metrics       # nom du job
#SBATCH --nodes=1                    # on demande 2 noeuds
#SBATCH --ntasks-per-node=1          # avec 2 tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=12           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --mem=100G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --partition=jean-zellou

#SBATCH -e /mnt/stores/store-DAI/tmp/speillet/logs/celest_metrics-%j.out
#SBATCH -o /mnt/stores/store-DAI/tmp/speillet/logs/celest_metrics-%j.out

cat "$0"
echo "========================================================================================"

export SRUN_CPUS_PER_TASK=12
export SBATCH_DEBUG=1
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1

# ==============================================================================
# NOUVELLES VARIABLES D'ENVIRONNEMENT POUR RASTERIO / DATALOADER
# ==============================================================================
# Empêche chaque worker PyTorch d'utiliser ses propres threads internes, 
# ce qui crée des blocages et des fuites mémoire avec Rasterio/GDAL
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Limite la mémoire cache interne de GDAL pour éviter de saturer la RAM
export GDAL_CACHEMAX=512
# ==============================================================================

set -x

eval "$(conda shell.bash hook)"
conda deactivate
conda activate cloud_reconstruction

# Créer un dossier temporaire dans votre espace personnel
export TMPDIR=/mnt/stores/store-DAI/tmp/speillet/torch_tmp
mkdir -p $TMPDIR

# Forcer Python et PyTorch à utiliser ce dossier
export TEMP=$TMPDIR
export TMP=$TMPDIR

srun --cpu-bind=none python /mnt/common/hdd/home/SPeillet/U-TILISE/metriques_celestin/metriques.py