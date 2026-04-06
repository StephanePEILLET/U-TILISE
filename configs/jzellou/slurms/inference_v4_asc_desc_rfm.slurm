#!/bin/bash
#SBATCH --job-name=infer_v4_ad_rfm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --partition=jean-zellou

#SBATCH -e /mnt/stores/store-DAI/tmp/speillet/logs/infer_v4_asc_desc_rfm-%j.out
#SBATCH -o /mnt/stores/store-DAI/tmp/speillet/logs/infer_v4_asc_desc_rfm-%j.out

cat "$0"
echo "================================================================================"

export SRUN_CPUS_PER_TASK=12
export SBATCH_DEBUG=1
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GDAL_CACHEMAX=512

set -x

eval "$(conda shell.bash hook)"
conda deactivate
conda activate cloud_reconstruction

export TMPDIR=/mnt/stores/store-DAI/tmp/speillet/torch_tmp
mkdir -p $TMPDIR
export TEMP=$TMPDIR
export TMP=$TMPDIR

WORK_DIR="/mnt/common/hdd/home/SPeillet/U-TILISE"
srun --cpu-bind=none python $WORK_DIR/run_inference.py $WORK_DIR/configs/jzellou/configs/config_run_inference_v4_asc_desc_rfm.yaml utilise
echo "Job complete"
