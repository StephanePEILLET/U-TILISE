#!/bin/bash
# Script pour lancer toutes les évaluations de métriques sur Jean Zellou
# Usage: bash configs/jzellou/slurms/launch_all_metrics.sh

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Lancement des évaluations de métriques ==="
echo "Répertoire SLURM: $SLURM_DIR"
echo ""

# all_bands_sar_asc_desc_random_clouds_da
sbatch "$SLURM_DIR/metrics_all_bands_sar_asc_desc_random_clouds_da_random_fully_masked.slurm"
sbatch "$SLURM_DIR/metrics_all_bands_sar_asc_desc_random_clouds_da_consecutive_fully_masked.slurm"

# all_bands_sar_asc_desc_random_clouds
sbatch "$SLURM_DIR/metrics_all_bands_sar_asc_desc_random_clouds_random_fully_masked.slurm"
sbatch "$SLURM_DIR/metrics_all_bands_sar_asc_desc_random_clouds_consecutive_fully_masked.slurm"

# all_bands_sar_closest_mix_random_fully_masked
sbatch "$SLURM_DIR/metrics_all_bands_sar_closest_mix_random_fully_masked_random_fully_masked.slurm"
sbatch "$SLURM_DIR/metrics_all_bands_sar_closest_mix_random_fully_masked_consecutive_fully_masked.slurm"

# all_bands_sar_closest_mix_random_fully_masked_da
sbatch "$SLURM_DIR/metrics_all_bands_sar_closest_mix_random_fully_masked_da_random_fully_masked.slurm"
sbatch "$SLURM_DIR/metrics_all_bands_sar_closest_mix_random_fully_masked_da_consecutive_fully_masked.slurm"

# all_bands_sar_asc_desc_random_fully_masked_da
sbatch "$SLURM_DIR/metrics_all_bands_sar_asc_desc_random_fully_masked_da_random_fully_masked.slurm"
sbatch "$SLURM_DIR/metrics_all_bands_sar_asc_desc_random_fully_masked_da_consecutive_fully_masked.slurm"

echo ""
echo "=== 10 jobs soumis ==="
