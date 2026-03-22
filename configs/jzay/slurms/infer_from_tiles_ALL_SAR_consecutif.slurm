#!/bin/bash
#SBATCH --job-name=infer_allsar_cons      # nom du job
#SBATCH --nodes=1                    # on demande 1 noeud
#SBATCH --ntasks-per-node=1          # avec 1 tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=36           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH -e /lustre/fsn1/worksf/projects/rech/tel/uug84ql/cloud_reconstruction/logs/infer_allsar_cons-%j.out
#SBATCH -o /lustre/fsn1/worksf/projects/rech/tel/uug84ql/cloud_reconstruction/logs/infer_allsar_cons-%j.out
#SBATCH --account=tel@v100

module purge

export SRUN_CPUS_PER_TASK=36
export SBATCH_DEBUG=1
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
export CLEARML_OFFLINE_MODE=1

module load python

conda deactivate
conda activate cloud_reconstruction

set -x

WORK_DIR="/lustre/fsn1/worksf/projects/rech/tel/uug84ql/cloud_reconstruction/U-TILISE"
srun --cpu-bind=none python $WORK_DIR/infer_from_tiles.py $WORK_DIR/configs/jzay/configs/config_run_infer_from_tiles_ALL_SAR_consecutif.yaml utilise
