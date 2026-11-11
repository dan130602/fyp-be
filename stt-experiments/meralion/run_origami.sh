#!/usr/bin/env bash
set -euo pipefail
umask 077
scripts="$(cd "$(dirname "$0")" && pwd)"
job="${1:?Usage: bash run_origami.sh JOB_ID energy|guarded 30|24 --all|--calls CALL_...}"
strategy="${2:?Specify energy or guarded}"
seconds="${3:?Specify 30 or 24}"
shift 3
[[ "$job" =~ ^[0-9]+$ && "$strategy" =~ ^(energy|guarded)$ && "$seconds" =~ ^(24|30)$ ]] || exit 2
[[ $# -gt 0 ]] || { echo 'Explicit --all or --calls required' >&2; exit 2; }
root="${MERALION_WORK:?Set MERALION_WORK to a private directory outside Git containing batch/ and model/}"
python="${MERALION_PYTHON:-python}"
allocation=$(squeue -h -j "$job" -o '%u %T %N')
[[ "$allocation" == "$USER RUNNING "* ]] || { echo 'Allocation is unavailable' >&2; exit 1; }
mkdir -p "$root/results"
exec 9>"$root/results/inference.lock"
flock -n 9 || { echo 'Another refinement is running in this directory' >&2; exit 1; }
cd "$root/model"
before="$root/results/$strategy-$seconds-weights.sha256"
sha256sum ./*.safetensors > "$before"
srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
  "$python" -B -u "$scripts/run_asr.py" --batch "$root/batch" --model "$root/model" \
  --strategy "$strategy" --maximum-seconds "$seconds" --output "$root/results/$strategy-$seconds" "$@" 2>&1 | \
  tee -a "$root/results/$strategy-$seconds.log"
sha256sum -c "$before"
