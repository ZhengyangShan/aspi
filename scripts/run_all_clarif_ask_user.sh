#!/usr/bin/env bash
# Run the clarif_ask_user evaluation for every model in a list, then build
# the ASR table. Re-running resumes: finished groups are kept, errored
# groups (e.g. rate-limited) are re-run.
#
# The proxy key's request limit is shared by every model, so keep
#   MAX_PARALLEL (models at once) x WORKERS (tasks per model)
# small enough that ~6 calls/task fit under it. At 120 requests/min,
# 3 x 3 = 9 concurrent tasks is about the ceiling.
#
# Usage:
#   bash scripts/run_all_clarif_ask_user.sh                       # full run
#   bash scripts/run_all_clarif_ask_user.sh --pilot               # 5 groups/suite
#   MAX_PARALLEL=3 WORKERS=3 PYTHON=python bash scripts/run_all_clarif_ask_user.sh
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHON="${PYTHON:-python}"
MODELS="${MODELS:-configs/clarif_ask_user_models.txt}"
export WORKERS="${WORKERS:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
export ATTACK="${ATTACK:-important_instructions}"
OUT="results/clarif_ask_user"
EXTRA=""
if [[ "${1:-}" == "--pilot" ]]; then
  OUT="results/clarif_ask_user_pilot"
  EXTRA="--sample-per-suite 5 --seed 0"
fi
# Keep a non-default attack's results in their own directory.
[[ "$ATTACK" != "important_instructions" ]] && OUT="${OUT}_${ATTACK}"
export OUT="${OUTPUT_DIR:-$OUT}" EXTRA
mkdir -p "$OUT/logs"
export DATASETS="data/aspi_banking_gemini-3.1-pro-preview.jsonl data/aspi_slack_gemini-3.1-pro-preview.jsonl data/aspi_travel_gemini-3.1-pro-preview.jsonl data/aspi_workspace_gemini-3.1-pro-preview.jsonl"

# One model per job, at most MAX_PARALLEL jobs at a time (xargs works with
# the bash 3.2 that ships with macOS, which has no `wait -n`).
grep -v -E '^\s*(#|$)' "$MODELS" | xargs -P "$MAX_PARALLEL" -I{} bash -c '
  model="$1"; log="$OUT/logs/$(echo "$model" | tr "/" "_").log"
  echo "starting $model -> $log"
  # shellcheck disable=SC2086
  "$PYTHON" scripts/run_clarif_ask_user.py --model "$model" --attack "$ATTACK" --dataset $DATASETS \
      --workers "$WORKERS" --output-dir "$OUT" --resume $EXTRA >> "$log" 2>&1
  echo "finished $model (exit $?)"
' _ {}

"$PYTHON" scripts/build_asr_table.py "$OUT"
