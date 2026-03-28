#!/bin/bash
# Script pour lancer les évaluations des modèles v3 variants sur Jean Zellou
# Usage: bash configs/jzellou/slurms/launch_metrics_v3_variants.sh

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Lancement des évaluations v3 variants ==="
echo "Répertoire SLURM: $SLURM_DIR"
echo ""

# v3 loss (L1 + SSIM + L1_occluded)
echo "--- v3 loss ---"
sbatch "$SLURM_DIR/metrics_v3_loss_rfm.slurm"
sbatch "$SLURM_DIR/metrics_v3_loss_cfm.slurm"

# v3 wider (architecture élargie)
echo "--- v3 wider ---"
sbatch "$SLURM_DIR/metrics_v3_wider_rfm.slurm"
sbatch "$SLURM_DIR/metrics_v3_wider_cfm.slurm"

# v3 combined (wider + cyclic + loss combinée)
echo "--- v3 combined ---"
sbatch "$SLURM_DIR/metrics_v3_combined_rfm.slurm"
sbatch "$SLURM_DIR/metrics_v3_combined_cfm.slurm"

# v3 cyclic (CosineAnnealingWarmRestarts) — attention: training divergé (NaN)
echo "--- v3 cyclic (divergé, test du best checkpoint) ---"
sbatch "$SLURM_DIR/metrics_v3_cyclic_rfm.slurm"
sbatch "$SLURM_DIR/metrics_v3_cyclic_cfm.slurm"

echo ""
echo "=== 8 jobs soumis ==="
