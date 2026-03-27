#!/bin/bash
# Launch remaining parcelle evaluations sequentially
# Waits for any running eval to finish first
set -e
cd /home/SPeillet/rpg_3STR/cloud_reconstruction/U-TILISE

LOG_DIR=/tmp
RUNNING_PID=$(pgrep -f "run_eval.py configs/config_run_eval_ALL_SAR_cfm_parcelle" 2>/dev/null | head -1 || true)

if [[ -n "$RUNNING_PID" ]]; then
    echo "$(date): Waiting for ALL_SAR cfm parcelle (PID $RUNNING_PID) to finish..."
    while kill -0 "$RUNNING_PID" 2>/dev/null; do
        sleep 30
    done
    echo "$(date): ALL_SAR cfm parcelle finished."
fi

echo "$(date): === [1/3] ALL_SAR rfm parcelle ==="
conda run -n cr --no-capture-output python run_eval.py configs/config_run_eval_ALL_SAR_rfm_parcelle.yaml utilise 2>&1 | tee "$LOG_DIR/eval_ALL_SAR_rfm_parcelle.log"
echo "$(date): ALL_SAR rfm parcelle done."

echo "$(date): === [2/3] v2_mix_rc rfm parcelle ==="
conda run -n cr --no-capture-output python run_eval.py configs/config_run_eval_v2_mix_rc_rfm_parcelle.yaml utilise 2>&1 | tee "$LOG_DIR/eval_v2_rfm_parcelle.log"
echo "$(date): v2 rfm parcelle done."

echo "$(date): === [3/3] v2_mix_rc cfm parcelle ==="
conda run -n cr --no-capture-output python run_eval.py configs/config_run_eval_v2_mix_rc_cfm_parcelle.yaml utilise 2>&1 | tee "$LOG_DIR/eval_v2_cfm_parcelle.log"
echo "$(date): v2 cfm parcelle done."

echo ""
echo "$(date): All remaining evaluations complete."
