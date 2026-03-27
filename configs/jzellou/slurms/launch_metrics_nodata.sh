#!/bin/bash
# Launch all 4 metric evaluation jobs (with nodata pixel filtering)
# Models: ALL_SAR_120_epochs + v2_mix_closest_random_clouds
# Masks:  random_fully_masked (rfm) + consecutive_fully_masked (cfm)

SLURM_DIR="$(dirname "$0")"

echo "Submitting ALL_SAR rfm..."
sbatch "$SLURM_DIR/metrics_ALL_SAR_rfm.slurm"

echo "Submitting ALL_SAR cfm..."
sbatch "$SLURM_DIR/metrics_ALL_SAR_cfm.slurm"

echo "Submitting v2_mix_rc rfm..."
sbatch "$SLURM_DIR/metrics_v2_mix_rc_rfm.slurm"

echo "Submitting v2_mix_rc cfm..."
sbatch "$SLURM_DIR/metrics_v2_mix_rc_cfm.slurm"

echo "All 4 jobs submitted."
