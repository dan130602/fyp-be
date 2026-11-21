#!/bin/bash
# Submit ONE SFT GPU job and record its allocation (run on Origami).   bash submit_sft.sh <script.sbatch> <blackwell|l40s|a40|3090> <HH:MM:SS> [KEY=VALUE ...]
# No-requeue (a preempted job is reported, never silently repeated); the allocation is appended to $SFT/gpu-ledger.tsv the moment sbatch returns. No cap is enforced here: the cap is the user's approval.
set -euo pipefail
SFT=/common/scratch/users/e/YOUR_USERNAME/meralion/sft
SCRIPT=${1:?script}; GPU=${2:?gpu}; LIMIT=${3:?time limit}; shift 3
case "$GPU" in
  blackwell) GRES=rtx_pro_6000_blackwell_max ;; l40s) GRES=l40s ;; a40) GRES=a40 ;; 3090) GRES=3090 ;;
  *) echo "unknown gpu type $GPU"; exit 2 ;;
esac
EXPORTS="ALL,GPU_TAG=$GPU"
for pair in "$@"; do EXPORTS="$EXPORTS,$pair"; done
mkdir -p "$SFT/logs"; cd "$SFT/logs"
JOB=$(sbatch --parsable --no-requeue --gres="gpu:$GRES:1" --constraint="$GRES" --time="$LIMIT" --export="$EXPORTS" "$SCRIPT")
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$JOB" "$(basename "$SCRIPT")" "$GPU" "$LIMIT" >> "$SFT/gpu-ledger.tsv"
echo "submitted job $JOB ($GPU, limit $LIMIT); ledger: $SFT/gpu-ledger.tsv"
