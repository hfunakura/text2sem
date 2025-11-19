#!/usr/bin/env bash
set -euo pipefail

export UV_HTTP_TIMEOUT=300

if [ $# -lt 4 ]; then
  echo "Usage: $0 \"<seeds>\" \"<tasks>\" \"<models>\" <script_path>"
  exit 1
fi

SEEDS=($1)
TASKS=($2)
MODELS=($3)
SCRIPT=$4

LOG_DIR="${PWD}/results/ICL_results"
mkdir -p "${LOG_DIR}"

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "=== Running: model=${model}, task=${task}, seed=${seed} ==="
      LOG_FILE="${LOG_DIR}/${model}_${task}_seed${seed}.log"

      uv run "${SCRIPT}" \
        --model "${model}" \
        --task "${task}" \
        --seed "${seed}" \
        --retries 3 \
        --backoff 2.0 \
        > "${LOG_FILE}" 2>&1

      echo "Log written to ${LOG_FILE}"
    done
  done
done
