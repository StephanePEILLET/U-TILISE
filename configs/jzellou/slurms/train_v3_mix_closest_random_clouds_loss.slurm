#!/bin/bash
#SBATCH --job-name=train_v3_loss         # nom du job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --partition=jean-zellou

#SBATCH -e /mnt/stores/store-DAI/tmp/speillet/logs/train_v3_mix_closest_random_clouds_loss-%j.out
#SBATCH -o /mnt/stores/store-DAI/tmp/speillet/logs/train_v3_mix_closest_random_clouds_loss-%j.out

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

srun --cpu-bind=none python /mnt/common/hdd/home/SPeillet/U-TILISE/run_train.py /mnt/common/hdd/home/SPeillet/U-TILISE/configs/jzellou/configs/train_v3_mix_closest_random_clouds_loss.yaml --save_dir /mnt/common/hdd/home/SPeillet/outputs/U-TILISE/v3_mix_closest_random_clouds_loss
echo "Job complete"
