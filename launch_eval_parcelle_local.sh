#!/bin/bash
# Launch all 4 parcel-only metric evaluations sequentially on this machine
# Usage: bash launch_eval_parcelle_local.sh
set -e
cd "$(dirname "$0")"

echo "=== [1/4] ALL_SAR rfm parcelle ==="
conda run -n cr --no-capture-output python run_eval.py configs/config_run_eval_ALL_SAR_rfm_parcelle.yaml utilise

echo "=== [2/4] ALL_SAR cfm parcelle ==="
conda run -n cr --no-capture-output python run_eval.py configs/config_run_eval_ALL_SAR_cfm_parcelle.yaml utilise

echo "=== [3/4] v2_mix_rc rfm parcelle ==="
conda run -n cr --no-capture-output python run_eval.py configs/config_run_eval_v2_mix_rc_rfm_parcelle.yaml utilise

echo "=== [4/4] v2_mix_rc cfm parcelle ==="
conda run -n cr --no-capture-output python run_eval.py configs/config_run_eval_v2_mix_rc_cfm_parcelle.yaml utilise

echo ""
echo "All 4 evaluations complete."
