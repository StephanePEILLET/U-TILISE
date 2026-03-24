#!/bin/bash
# Lance les métriques du modèle LEGACY ALL_SAR_120_epochs sur Jean Zellou
# Ce modèle a été entraîné avec un dataloader qui masquait aussi les données SAR (erreur).
# Pour reproduire les conditions d'entraînement lors de l'évaluation :
#   - mask_sar: true → SAR masqué comme pendant l'entraînement (mode legacy)
#   - max_seq_length: 10 → cohérent avec l'entraînement
# Note : les nouveaux modèles utilisent mask_sar=false (comportement correct, SAR non masqué)
# 2 modes de masquage : aléatoire (random_fully_masked) et consécutif (consecutive_fully_masked)
# Usage: bash configs/jzellou/slurms/launch_metrics_ALL_SAR.sh

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Métriques modèle LEGACY ALL_SAR_120_epochs_2025-07-11_16-56 ==="
echo "  HDF5 : /var/data/datasets/CIRCA_CR_merged.hdf5"
echo "  Checkpoint : /mnt/stores/store-DAI/tmp/speillet/cloud_reconstruction_results/U-TILISE/ALL_SAR_120_epochs_2025-07-11_16-56/"
echo "  Mode legacy : mask_sar=true, max_seq_length=10"
echo ""

echo "  Soumission masques ALÉATOIRES (random_fully_masked)..."
sbatch "$SLURM_DIR/metrics_ALL_SAR_random_fully_masked.slurm"

echo "  Soumission masques CONSÉCUTIFS (consecutive_fully_masked)..."
sbatch "$SLURM_DIR/metrics_ALL_SAR_consecutive_fully_masked.slurm"

echo ""
echo "=== 2 jobs soumis. Vérifiez avec: squeue -u \$USER ==="
echo "=== Logs v2 : /mnt/stores/store-DAI/tmp/speillet/logs/metrics_allsar_*_v2-*.out ==="
