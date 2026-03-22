#!/bin/bash
# Lance les inférences par tuiles avec masques synthétiques sur Jean Zay
# 3 checkpoints x 2 modes de masquage = 6 jobs
# Usage: bash launch_infer_masks_jzay.sh

set -e

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/configs/jzay/slurms"

echo "=== Checkpoint: all_bands_sar_closest_mix_da ==="
echo "  Soumission masques ALÉATOIRES..."
sbatch "$SLURM_DIR/infer_from_tiles_aleatoire.slurm"
echo "  Soumission masques CONSÉCUTIFS..."
sbatch "$SLURM_DIR/infer_from_tiles_consecutif.slurm"

echo ""
echo "=== Checkpoint: ALL_SAR_120_epochs ==="
echo "  Soumission masques ALÉATOIRES..."
sbatch "$SLURM_DIR/infer_from_tiles_ALL_SAR_aleatoire.slurm"
echo "  Soumission masques CONSÉCUTIFS..."
sbatch "$SLURM_DIR/infer_from_tiles_ALL_SAR_consecutif.slurm"

echo ""
echo "=== Checkpoint: all_bands_sar_closest_mix_random_fully_masked_da ==="
echo "  Soumission masques ALÉATOIRES..."
sbatch "$SLURM_DIR/infer_from_tiles_rfm_da_aleatoire.slurm"
echo "  Soumission masques CONSÉCUTIFS..."
sbatch "$SLURM_DIR/infer_from_tiles_rfm_da_consecutif.slurm"

echo ""
echo "6 jobs soumis. Vérifiez avec: squeue -u \$USER"
