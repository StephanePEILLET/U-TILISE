#!/bin/bash
#SBATCH --job-name=m_v2rc_rfm_parc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --partition=jean-zellou

#SBATCH -e /mnt/stores/store-DAI/tmp/speillet/logs/m_v2rc_rfm_parc-%j.out
#SBATCH -o /mnt/stores/store-DAI/tmp/speillet/logs/m_v2rc_rfm_parc-%j.out

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

srun python /mnt/common/hdd/home/SPeillet/U-TILISE/run_eval.py /mnt/common/hdd/home/SPeillet/U-TILISE/configs/jzellou/configs/config_run_eval_v2_mix_rc_storeDAI_rfm_parcelle.yaml utilise
echo "Job complete"
