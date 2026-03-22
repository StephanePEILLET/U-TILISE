#!/bin/bash
# Lance les inférences par tuiles avec masques synthétiques (aléatoire et consécutif)
# Usage: bash launch_infer_masks.sh [aleatoire|consecutif|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-all}"

run_aleatoire() {
    echo "============================================="
    echo " Inférence avec masques ALÉATOIRES"
    echo "============================================="
    python infer_from_tiles.py configs/config_run_infer_from_tiles_aleatoire.yaml utilise
}

run_consecutif() {
    echo "============================================="
    echo " Inférence avec masques CONSÉCUTIFS"
    echo "============================================="
    python infer_from_tiles.py configs/config_run_infer_from_tiles_consecutif.yaml utilise
}

case "$MODE" in
    aleatoire)
        run_aleatoire
        ;;
    consecutif)
        run_consecutif
        ;;
    all)
        run_aleatoire
        echo ""
        run_consecutif
        ;;
    *)
        echo "Usage: $0 [aleatoire|consecutif|all]"
        exit 1
        ;;
esac

echo ""
echo "Terminé."
