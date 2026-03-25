#!/bin/bash
# Launch all v2 metrics evaluation jobs on jean-zellou
# Usage: bash configs/jzellou/slurms/launch_metrics_v2.sh

SLURM_DIR="$(dirname "$0")"

echo "Submitting v2 metrics jobs..."

# v2 mix_closest, random_fully_masked
sbatch "$SLURM_DIR/metrics_v2_mix_closest_random_fully_masked_rfm.slurm"
sbatch "$SLURM_DIR/metrics_v2_mix_closest_random_fully_masked_cfm.slurm"

# v2 mix_closest, random_clouds
sbatch "$SLURM_DIR/metrics_v2_mix_closest_random_clouds_rfm.slurm"
sbatch "$SLURM_DIR/metrics_v2_mix_closest_random_clouds_cfm.slurm"

# v2 asc+desc, random_fully_masked
sbatch "$SLURM_DIR/metrics_v2_asc_desc_random_fully_masked_rfm.slurm"
sbatch "$SLURM_DIR/metrics_v2_asc_desc_random_fully_masked_cfm.slurm"

# v2 asc+desc, random_clouds
sbatch "$SLURM_DIR/metrics_v2_asc_desc_random_clouds_rfm.slurm"
sbatch "$SLURM_DIR/metrics_v2_asc_desc_random_clouds_cfm.slurm"

echo "All 8 v2 metrics jobs submitted."
