#!/bin/bash
# GLUE baseline runs: PackR + Velvet Auto-tune (VA)
# Sequential execution — one GPU. Auto-resumes by skipping completed runs.
# All GLUE tasks at 3 epochs (matching Devlin et al. 2019 full finetune baseline).
set -euo pipefail

cd "$(dirname "$0")"
source .venv-f39/bin/activate

mkdir -p logs results

TASKS=("wnli" "rte" "mrpc" "cola" "stsb" "sst2" "qnli" "mnli" "qqp")

# Standard GLUE fine-tuning: 3 epochs (Devlin et al. 2019)
declare -A EPOCHS=(
  ["cola"]=3  ["mrpc"]=3  ["qqp"]=3  ["qnli"]=3
  ["rte"]=3   ["stsb"]=3  ["sst2"]=3 ["mnli"]=3 ["wnli"]=3
)

METHOD="packr_va"
CURRENT_COMMIT=$(git rev-parse --short HEAD)

has_results() {
  local task="$1" seed="${2:-42}"
  for d in results/${task}_${METHOD}_seed${seed}_*; do
    [ -d "$d" ] || continue
    if [ -f "$d/metrics.json" ]; then
      local dcommit
      dcommit=$(basename "$d" | grep -oP "(?<=seed${seed}_)[a-f0-9]+" || echo "")
      if [ "$dcommit" != "$CURRENT_COMMIT" ]; then
        echo "  [STALE] Removing $task result from commit $dcommit (current: $CURRENT_COMMIT)"
        rm -rf "$d"
        continue
      fi
      local train_acc
      train_acc=$(python3 -c "import json; d=json.load(open('$d/metrics.json')); print(d.get('train_acc', 0))" 2>/dev/null || echo "0")
      if [ "$train_acc" != "0" ] && [ "$train_acc" != "0.0" ]; then
        local val_acc
        val_acc=$(python3 -c "import json; d=json.load(open('$d/metrics.json')); print(d.get('val_acc', d.get('val_corr', 0)))" 2>/dev/null || echo "0")
        echo "  [SKIP] $METHOD/$task — valid results exist (val=$val_acc%, train=$train_acc%)"
        return 0
      else
        echo "  [CLEAN] Removing broken run: $d"
        rm -rf "$d"
      fi
    fi
  done
  return 1
}

for task in "${TASKS[@]}"; do
  epochs="${EPOCHS[$task]}"
  echo ""
  echo "================================================================"
  echo "  TASK: $task  (epochs=$epochs)"
  echo "================================================================"

  if has_results "$task"; then
    continue
  fi

  logfile="logs/${task}_${METHOD}_$(date +%Y%m%d_%H%M%S).log"
  echo ""
  echo "--- $(date '+%H:%M:%S')  $METHOD / $task ---"
  echo "  Log: $logfile"
  echo "  Cmd: python -m tests.harness --method=packr --velvet --task=$task --epochs=$epochs"

  python -m tests.harness --method=packr --velvet --task="$task" --epochs="$epochs" \
    2>&1 | tee "$logfile"

  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "  [FAILED] $METHOD/$task exited with code $exit_code"
  fi
  echo "--- $(date '+%H:%M:%S') done ---"
done

echo ""
echo "================================================================"
echo "  ALL PACKR+VA RUNS COMPLETE"
echo "================================================================"
echo ""
echo "  Baseline: bert-base-uncased full finetune, 3 epochs (Devlin et al. 2019)"
echo "  PackR+VA runs at 3 epochs — equal compute, lower VRAM."
echo ""
echo "  Task   | Full FT | PackR+VA"
echo "  -------|---------|----------"
echo "  CoLA   | 52.1    |"
echo "  SST-2  | 92.7    |"
echo "  MRPC   | 88.9    |"
echo "  STS-B  | 87.1    |"
echo "  QQP    | 87.5    |"
echo "  MNLI   | 84.6    |"
echo "  QNLI   | 90.5    |"
echo "  RTE    | 66.4    |"
echo "  WNLI   | 56.3    |"
echo ""
echo "  Results: results/"
echo "  Logs:    logs/"
echo ""
echo "  Run analyzer: python -m tests.analyzer"
