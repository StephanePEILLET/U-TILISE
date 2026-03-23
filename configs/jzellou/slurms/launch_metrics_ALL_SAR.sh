#!/bin/bash
# Lance les métriques du modèle ALL_SAR_120_epochs sur Jean Zellou
# 2 modes de masquage : aléatoire (random_fully_masked) et consécutif (consecutive_fully_masked)
# Usage: bash configs/jzellou/slurms/launch_metrics_ALL_SAR.sh

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Métriques ALL_SAR_120_epochs_2025-07-11_16-56 ==="
echo "  HDF5 : /var/data/datasets/CIRCA_CR_merged.hdf5"
echo "  Checkpoint : /mnt/stores/store-DAI/tmp/speillet/cloud_reconstruction_results/U-TILISE/ALL_SAR_120_epochs_2025-07-11_16-56/"
echo ""

echo "  Soumission masques ALÉATOIRES (random_fully_masked)..."
sbatch "$SLURM_DIR/metrics_ALL_SAR_random_fully_masked.slurm"

echo "  Soumission masques CONSÉCUTIFS (consecutive_fully_masked)..."
sbatch "$SLURM_DIR/metrics_ALL_SAR_consecutive_fully_masked.slurm"

echo ""
echo "=== 2 jobs soumis. Vérifiez avec: squeue -u \$USER ==="
