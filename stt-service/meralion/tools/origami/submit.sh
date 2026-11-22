#!/bin/bash
# Submit one follow-up job with the GPU type and time limit given on the sbatch COMMAND LINE (shell variables are never expanded inside #SBATCH lines), after checking the GPU-time ledger.
# The allocated limit is recorded in full the moment sbatch returns, even if the job fails, is cancelled or ends early; a job whose limit exceeds the remaining allowance is refused.
#   bash submit.sh <script.sbatch> <gpu: blackwell|l40s|a40|3090|2080> <HH:MM:SS> [KEY=VALUE ...]      (KEY=VALUE pairs are exported to the job)
# Every job goes through sbatch --no-requeue (REV 3.0 A2: an acceptance run is one uninterrupted attempt; under PreemptMode=REQUEUE a preempted non-requeueable job is cancelled, which is
# reported, never silently repeated). Order, with every step handled explicitly:
#   1. ledger check            2. sbatch --no-requeue (a failure submits and records nothing)            3. ledger record IMMEDIATELY (a write failure: scancel, exit 5)
#   4. scontrol show job must show Requeue=0; a failed or empty inspection, a missing field or Requeue=1 is a FAILURE: scancel, annotate the ledger entry as cancelled (it still counts), exit 6.
# W0_BASE overrides the base folder (tests only).
set -euo pipefail
SCR=/common/scratch/users/e/YOUR_USERNAME/meralion
BASE=${W0_BASE:-$SCR/runs/worker-v0}
CODE=$BASE/code
LEDGER=$BASE/worker-v0-gpu-ledger.json
CAP_HOURS=${GPU_CAP_HOURS:-5.0}
SCRIPT=${1:?script}
GPU=${2:?gpu}
LIMIT=${3:?time limit HH:MM:SS}
shift 3
case "$GPU" in
  blackwell) GRES=rtx_pro_6000_blackwell_max; FEATURE=rtx_pro_6000_blackwell_max ;;
  l40s)      GRES=l40s;  FEATURE=l40s ;;
  a40)       GRES=a40;   FEATURE=a40 ;;
  3090)      GRES=3090;  FEATURE=3090 ;;
  2080)      GRES=2080;  FEATURE=2080 ;;
  *) echo "unknown gpu type $GPU (blackwell, l40s, a40, 3090, 2080)"; exit 2 ;;
esac
EXPORTS="ALL,GPU_TAG=$GPU"
for pair in "$@"; do EXPORTS="$EXPORTS,$pair"; done
python3 "$CODE/tools/origami/ledger.py" check "$LEDGER" "$CAP_HOURS" "$LIMIT"
mkdir -p "$BASE/logs"
cd "$BASE/logs"
JOB=$(sbatch --parsable --no-requeue --gres="gpu:$GRES:1" --constraint="$FEATURE" --time="$LIMIT" --export="$EXPORTS" "$SCRIPT") || { echo "sbatch failed: nothing was submitted and nothing was recorded"; exit 4; }
if ! python3 "$CODE/tools/origami/ledger.py" record "$LEDGER" "$CAP_HOURS" "$LIMIT" "$JOB" "$(basename "$SCRIPT")" "$GPU"; then
  scancel "$JOB" 2>/dev/null || true
  echo "LEDGER WRITE FAILED: job $JOB with limit $LIMIT is submitted or cancelled but NOT recorded; add it to $LEDGER by hand (the allocation counts in full)"
  exit 5
fi
echo "submitted job $JOB ($GPU, limit $LIMIT)"
INFO=""
INFO_RC=0
INFO=$(scontrol show job "$JOB" 2>&1) || INFO_RC=$?
REASON=""
if [ "$INFO_RC" != "0" ] || [ -z "$INFO" ]; then
  REASON="requeue state could not be verified: scontrol show job failed or returned nothing (exit $INFO_RC)"
elif printf '%s\n' "$INFO" | grep -Eq '(^|[[:space:]])Requeue=0([[:space:]]|$)'; then
  REASON=""
elif printf '%s\n' "$INFO" | grep -Eq '(^|[[:space:]])Requeue=1([[:space:]]|$)'; then
  REASON="Requeue=1 despite --no-requeue"
else
  REASON="requeue state could not be verified: scontrol output has no Requeue field"
fi
if [ -n "$REASON" ]; then
  scancel "$JOB" 2>/dev/null || true
  python3 "$CODE/tools/origami/ledger.py" annotate "$LEDGER" "$JOB" cancelled "$REASON" || echo "WARNING: the ledger entry of job $JOB could not be annotated"
  STATE=$(squeue -h -j "$JOB" -o '%T' 2>/dev/null || true)
  echo "job $JOB CANCELLED: $REASON; queue state now: ${STATE:-not in the queue}; its allocation of $LIMIT stays counted in the ledger"
  exit 6
fi
python3 "$CODE/tools/origami/ledger.py" show "$LEDGER" "$CAP_HOURS" | head -4
