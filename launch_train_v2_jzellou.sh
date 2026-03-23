#!/bin/bash
# Lance les 4 entraînements v2 sur Jean Zellou
# Usage: bash launch_train_v2_jzellou.sh

SLURM_DIR="configs/jzellou/slurms"

echo "=== Soumission des 4 entraînements v2 ==="

echo "[1/4] mix_closest + random_clouds"
sbatch "$SLURM_DIR/train_v2_mix_closest_random_clouds.slurm"

echo "[2/4] mix_closest + random_fully_masked"
sbatch "$SLURM_DIR/train_v2_mix_closest_random_fully_masked.slurm"

echo "[3/4] asc+desc + random_clouds"
sbatch "$SLURM_DIR/train_v2_asc_desc_random_clouds.slurm"

echo "[4/4] asc+desc + random_fully_masked"
sbatch "$SLURM_DIR/train_v2_asc_desc_random_fully_masked.slurm"

echo ""
echo "=== Tous les jobs ont été soumis ==="
echo "Suivi : squeue -u \$USER"
