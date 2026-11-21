#!/bin/bash -l
# The Blackwell MIG-slice smoke of the BF16 profile vllm-rp105-sched0-v1 (REV 3.0 A1): the frozen configuration, nothing tuned. Every step is its own process; the FIRST failed step ends the job (the result is
# reported, never repaired by changing a setting: no concurrency, memory-fraction, dtype, backend or batch change is tried). One uninterrupted attempt (--no-requeue, require_first_attempt).
#   1 device record   2 apply_patches --check (state must be patched)   3 device_smoke on CALL_064 (4 windows)   4 vllm_windows with --capture on the 9 diagnostic windows
#   5 the worker on CALL_064 (short end-to-end)   6 the worker on CALL_103 alone (125 first-pass windows in one batch: the real memory and concurrency test)
# Functional equivalence of the adapter is established ONLY by tools/capture_replay.py on the laptop from step 4's capture. The controlled logits test is NOT run here (it runs on the login node, REV 3.2).
# Submit ONLY through submit.sh (type and time on the command line, ledger checked first; the limit for this job is 00:30:00):
#   GPU_CAP_HOURS=<approved cap> bash $CODE/tools/origami/submit.sh $CODE/tools/origami/slice_smoke.sbatch blackwell 00:30:00
# No engine environment variable is set anywhere in this script: the profile sets its own.
#SBATCH --job-name=w0-slice-smoke
#SBATCH --partition=student
#SBATCH --account=student
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_pro_6000_blackwell_max
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.log

set -eo pipefail
module load Python/3.12.8-GCCcore-13.3.0
module load CUDA/12.8.0
set -u
source /common/scratch/users/e/YOUR_USERNAME/meralion/runs/worker-v0/code/tools/origami/common.sh
: "${GPU_TAG:?GPU_TAG is set by submit.sh}"
export CUDA_HOME=${EBROOTCUDA:?CUDA module not loaded}
use_vllm_env
require_clean_commit
cd "$CODE"
OUT=$BASE/slice-smoke-$GPU_TAG-$SLURM_JOB_ID
require_first_attempt "$OUT"
mkdir -p "$OUT"
record_device "$OUT/device.txt"
cat "$OUT/device.txt"
SETTINGS_ID=vllm-rp105-sched0-v1
CONFIG=$CODE/configs/runtime/fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0.json
TOOLS=$CODE/tools/origami
JOBS_DEV=$BASE/jobs/jobs-dev175.json
JOBS_C4=$BASE/jobs/jobs-c4.json
JOBS_103=$OUT/jobs-call103.json
WINDOWS=$TOOLS/problem_windows.json
RUN_C4="vllm-$GPU_TAG-slice-c4"
RUN_103="vllm-$GPU_TAG-slice-call103"
test -f "$JOBS_DEV" && test -f "$JOBS_C4" && test -f "$CONFIG" && test -f "$WINDOWS"
for inv in "$RUN_C4" "$RUN_103"; do
  if [ -e "$BASE/runs/$inv/wall.json" ] || [ -e "$BASE/runs/$inv/run-meta.json" ]; then echo "run $inv already started: refusing to reuse its id"; exit 3; fi
done
# the CALL_103 jobs file is written from the 175-recording jobs file (the largest recording)
"$VENV_VLLM/bin/python" - "$JOBS_DEV" "$JOBS_103" <<'PYEOF'
import json, sys
source, target = sys.argv[1:3]
jobs = [j for j in json.load(open(source))["jobs"] if j["job_name"] == "CALL_103"]
if len(jobs) != 1:
    raise SystemExit(f"CALL_103 must be exactly once in {source}")
json.dump({"jobs": jobs}, open(target, "w"), indent=1)
PYEOF
CALL=$(python3 -c "import json; print(json.load(open('$JOBS_C4'))['jobs'][0]['job_name'])")
echo "slice smoke: settings $SETTINGS_ID, call $CALL, device $(allocated_identity)"
# a required step: its output is kept in $OUT/<name>.log, its exit code decides, nothing after a failed step runs, and a failure is classified (memory / software / numerical) for the report
need() {
  local name=$1
  shift
  echo "=== step $name"
  set +e
  "$@" 2>&1 | tee "$OUT/$name.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "=== step $name exit $rc"
  echo "$name $rc" >> "$OUT/steps.txt"
  if [ "$rc" != "0" ]; then
    echo "required step $name failed (exit $rc): nothing after it runs"
    "$VENV_VLLM/bin/python" "$TOOLS/device_smoke.py" classify "$OUT/$name.log" "$rc" "$OUT/device-smoke.json" || true
    exit "$rc"
  fi
}
need 2_apply_patches_check "$VENV_VLLM/bin/python" "$CODE/tools/apply_patches.py" --check --expect patched
need 3_device_smoke "$VENV_VLLM/bin/python" -u "$TOOLS/device_smoke.py" --settings-id "$SETTINGS_ID" --model-dir "$MODEL" --jobs "$JOBS_C4" --call "$CALL" --windows 4 \
    --expect-capability 12.0 --expect-platform NonNvmlCudaPlatform --out "$OUT/device-smoke.json"
need 4_windows_capture "$VENV_VLLM/bin/python" -u "$TOOLS/vllm_windows.py" --settings-id "$SETTINGS_ID" --jobs "$JOBS_DEV" --windows "$WINDOWS" --passes batch,batch,single \
    --capture "$OUT/capture.json" --out "$OUT/windows.json"
need 5_worker_call064 run_worker "$VENV_VLLM/bin/python" vllm "$RUN_C4" "$JOBS_C4" --config "$CONFIG"
need 6_worker_call103 run_worker "$VENV_VLLM/bin/python" vllm "$RUN_103" "$JOBS_103" --config "$CONFIG"
echo "slice smoke results in $OUT and $BASE/runs/$RUN_C4 and $BASE/runs/$RUN_103"
