#!/bin/bash
# Auto-resuming training supervisor: relaunches `geoshield.train --resume`
# from the last checkpoint if the process dies for any reason (crash, OOM,
# a laptop shutdown, or the environment killing background work — the
# post_only run died once mid-session with no Python traceback, most
# likely from exactly this). Runs post_only to completion, then siamese.
#
# Usage: ./ml/train_supervisor.sh
# Safe to stop (Ctrl+C, or just shut down the machine) and re-run later —
# it picks up from the last per-epoch checkpoint instead of restarting.
set -uo pipefail
cd "$(dirname "$0")/.."

REPO="$(pwd)"
OUTPUT=$REPO/ml/checkpoints
DATA=$REPO/data/prepared
MANIFEST=$REPO/data/prepared/split_manifest.json
EPOCHS=30
MAX_RETRIES=20

train_one() {
  local model=$1
  local retries=0
  while true; do
    completed=$(.venv/bin/python -c "
import json, os
p = '$OUTPUT/${model}_history.json'
print(len(json.load(open(p))) if os.path.exists(p) else 0)
")
    if [ "$completed" -ge "$EPOCHS" ]; then
      echo "$(date '+%F %T') [$model] already has $completed/$EPOCHS epochs recorded; treating as complete."
      return 0
    fi
    resume_args=()
    if [ -f "$OUTPUT/${model}_last.pt" ]; then
      resume_args=(--resume "$OUTPUT/${model}_last.pt")
    fi
    echo "$(date '+%F %T') [$model] launching (completed=$completed, retry=$retries, resume=${resume_args[*]:-none})"
    PYTHONPATH=ml .venv/bin/python -m geoshield.train \
      --model "$model" \
      --data "$DATA" \
      --split-manifest "$MANIFEST" \
      --output "$OUTPUT" \
      --epochs "$EPOCHS" \
      "${resume_args[@]}"
    code=$?
    echo "$(date '+%F %T') [$model] exited with code $code"
    if [ "$code" -eq 0 ]; then
      # Clean exit: either finished all epochs or early-stopped (both are
      # legitimate completions of train()'s loop, not failures).
      echo "$(date '+%F %T') [$model] clean exit, done."
      return 0
    fi
    retries=$((retries + 1))
    if [ "$retries" -ge "$MAX_RETRIES" ]; then
      echo "$(date '+%F %T') [$model] giving up after $retries retries."
      return 1
    fi
    sleep 5
  done
}

train_one post_only && train_one siamese
echo "$(date '+%F %T') supervisor finished all models."
