#!/bin/bash
#SBATCH --job-name=metrics_v3_combined_cfm        # nom du job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --partition=jean-zellou

#SBATCH -e /mnt/stores/store_dai/tmp/speillet/logs/metrics_v3_combined_cfm-%j.out
#SBATCH -o /mnt/stores/store_dai/tmp/speillet/logs/metrics_v3_combined_cfm-%j.out

cat "$0"
echo "================================================================================"

export SRUN_CPUS_PER_TASK=12
export SBATCH_DEBUG=1
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1

set -x

eval "$(conda shell.bash hook)"
conda deactivate
conda activate cloud_reconstruction

srun python /mnt/common/hdd/home/SPeillet/U-TILISE/run_eval.py /mnt/common/hdd/home/SPeillet/U-TILISE/configs/jzellou/configs/config_run_eval_v3_combined_cfm.yaml utilise
echo "Job complete"
