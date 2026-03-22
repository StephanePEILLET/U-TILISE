#!/bin/bash
# Script pour lancer les évaluations du modèle coherence_only sur Jean Zellou
# Usage: bash configs/jzellou/slurms/launch_metrics_coherence_only.sh

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Lancement des métriques coherence_only ==="

sbatch "$SLURM_DIR/metrics_coherence_only_random_fully_masked.slurm"
sbatch "$SLURM_DIR/metrics_coherence_only_consecutive_fully_masked.slurm"

echo "=== 2 jobs soumis ==="
