#!/bin/bash
# Launch metrics evaluation jobs (parcelle RPG only) for ALL_SAR + v2
# Usage: bash configs/jzellou/slurms/launch_metrics_parcelle.sh

SLURM_DIR="$(dirname "$0")"

echo "============================================="
echo " Submitting metrics jobs (parcelle RPG only) "
echo "============================================="

echo ""
echo "--- ALL_SAR_120_epochs ---"

echo "  [1/4] ALL_SAR rfm + parcelle"
sbatch "$SLURM_DIR/metrics_ALL_SAR_rfm_parcelle.slurm"

echo "  [2/4] ALL_SAR cfm + parcelle"
sbatch "$SLURM_DIR/metrics_ALL_SAR_cfm_parcelle.slurm"

echo ""
echo "--- v2_mix_closest_random_clouds (storeDAI) ---"

echo "  [3/4] v2 rfm + parcelle"
sbatch "$SLURM_DIR/metrics_v2_mix_rc_storeDAI_rfm_parcelle.slurm"

echo "  [4/4] v2 cfm + parcelle"
sbatch "$SLURM_DIR/metrics_v2_mix_rc_storeDAI_cfm_parcelle.slurm"

echo ""
echo "All 4 metrics jobs submitted."
echo ""
echo "Output directories:"
echo "  /mnt/common/hdd/home/SPeillet/outputs/U-TILISE/metrics/ALL_SAR_120_epochs/*_parcelle/"
echo "  /mnt/common/hdd/home/SPeillet/outputs/U-TILISE/metrics/v2_mix_closest_random_clouds_storeDAI/*_parcelle/"
