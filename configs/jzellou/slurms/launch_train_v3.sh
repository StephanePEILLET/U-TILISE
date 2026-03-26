#!/bin/bash
# Lance les 4 variantes d'entraînement v3
# (le v3 de base est déjà en cours, job 27714)

set -e

echo "=== Lancement des entraînements v3 sur jean-zellou ==="

echo "[1/4] v3 cyclic (CosineAnnealingWarmRestarts + gradient clipping)"
sbatch configs/jzellou/slurms/train_v3_mix_closest_random_clouds_cyclic.slurm

echo "[2/4] v3 loss (L1 + SSIM + L1 occluded w=2.0)"
sbatch configs/jzellou/slurms/train_v3_mix_closest_random_clouds_loss.slurm

echo "[3/4] v3 wider (encoder/decoder élargis, 8 heads)"
sbatch configs/jzellou/slurms/train_v3_mix_closest_random_clouds_wider.slurm

echo "[4/4] v3 combined (wider + cyclic + loss combinée)"
sbatch configs/jzellou/slurms/train_v3_mix_closest_random_clouds_combined.slurm

echo ""
echo "=== 4 jobs soumis. Vérifier avec : squeue -u \$USER ==="
