#!/bin/bash
#SBATCH --job-name=metrics_utilise_consecutive_fully_masked_random_clouds       # nom du job
#SBATCH --nodes=1                    # on demande 2 noeuds
#SBATCH --ntasks-per-node=1          # avec 2 tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=12           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --mem=100G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --partition=jean-zellou

#SBATCH -e /mnt/stores/store-DAI/tmp/speillet/logs/metrics_utilise_consecutive_fully_masked_random_clouds-%j.out
#SBATCH -o /mnt/stores/store-DAI/tmp/speillet/logs/metrics_utilise_consecutive_fully_masked_random_clouds-%j.out

cat "$0"
echo "========================================================================================"

export SRUN_CPUS_PER_TASK=12
export SBATCH_DEBUG=1
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1

set -x

eval "$(conda shell.bash hook)"
conda deactivate
conda activate cloud_reconstruction

srun python /mnt/common/hdd/home/SPeillet/U-TILISE/run_eval.py /mnt/common/hdd/home/SPeillet/U-TILISE/configs/jzellou/configs/config_run_eval_consecutive_fully_masked_random_clouds.yaml utilise
echo "Job complete"
