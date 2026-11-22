#!/bin/bash -l
# T0 (REV 3.0 B4, REV 3.1 Amendment 3): the SOFTWARE precursor of the T4 deployment profile vllm-rp105-sched0-fp16-v1 on one of the cluster's Turing GPUs (a whole RTX 2080 class board, compute capability 7.5).
# What it can show: the FP16 software path (FlashInfer JIT for sm75, the engine, decodes) works on a Turing board. What it cannot show: anything about a T4's memory or the EC2 host (this board has a
# different amount of memory, driver and host), so a memory-driven failure here is INCONCLUSIVE for the T4 and a pass is evidence for the software path only. The REAL device model and memory are measured
# and recorded; nothing assumes them. Nothing is tuned and nothing escalates: no dtype, memory fraction, concurrency, backend or package change is tried after a failure.
# The profile is named explicitly (there is no fallback from the BF16 profile); no patch is needed or checked on a whole GPU (a numeric device value keeps the unpatched platform selection).
#   1 device record + driver CUDA level >= 12.8   2 device_smoke (FP16, CALL_064, 4 windows, no empty text)   3 vllm_windows with --capture   4 the worker on CALL_064   5 the worker on CALL_103
# Submit ONLY through submit.sh (the limit for this job is 00:30:00):
#   GPU_CAP_HOURS=<approved cap> bash $CODE/tools/origami/submit.sh $CODE/tools/origami/fp16_smoke.sbatch 2080 00:30:00
#SBATCH --job-name=w0-fp16-smoke
#SBATCH --partition=student
#SBATCH --account=student
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080
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
OUT=$BASE/fp16-smoke-$GPU_TAG-$SLURM_JOB_ID
require_first_attempt "$OUT"
mkdir -p "$OUT"
record_device "$OUT/device.txt"
cat "$OUT/device.txt"
SETTINGS_ID=vllm-rp105-sched0-fp16-v1
CONFIG=$CODE/configs/runtime/fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0-fp16.json
TOOLS=$CODE/tools/origami
JOBS_DEV=$BASE/jobs/jobs-dev175.json
JOBS_C4=$BASE/jobs/jobs-c4.json
JOBS_103=$OUT/jobs-call103.json
WINDOWS=$TOOLS/problem_windows.json
RUN_C4="vllm-$GPU_TAG-fp16smoke-c4"
RUN_103="vllm-$GPU_TAG-fp16smoke-call103"
test -f "$JOBS_DEV" && test -f "$JOBS_C4" && test -f "$CONFIG" && test -f "$WINDOWS"
for inv in "$RUN_C4" "$RUN_103"; do
  if [ -e "$BASE/runs/$inv/wall.json" ] || [ -e "$BASE/runs/$inv/run-meta.json" ]; then echo "run $inv already started: refusing to reuse its id"; exit 3; fi
done
"$VENV_VLLM/bin/python" - "$JOBS_DEV" "$JOBS_103" <<'PYEOF'
import json, sys
source, target = sys.argv[1:3]
jobs = [j for j in json.load(open(source))["jobs"] if j["job_name"] == "CALL_103"]
if len(jobs) != 1:
    raise SystemExit(f"CALL_103 must be exactly once in {source}")
json.dump({"jobs": jobs}, open(target, "w"), indent=1)
PYEOF
CALL=$(python3 -c "import json; print(json.load(open('$JOBS_C4'))['jobs'][0]['job_name'])")
# the measured device (model, total memory, driver, compute capability) is recorded BEFORE anything runs: it is what the result is read against
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv | tee "$OUT/gpu-query.csv"
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
# the driver must support the CUDA 12.8 runtime of the pinned torch (read from nvidia-smi's own header, not assumed)
need 1_driver_supports_cuda_12_8 python3 - <<'PYEOF'
import re, subprocess, sys
header = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", header)
if not match:
    sys.exit("nvidia-smi shows no CUDA Version")
level = (int(match.group(1)), int(match.group(2)))
print("driver CUDA level", level)
sys.exit(0 if level >= (12, 8) else "the driver supports only CUDA %d.%d (the pinned torch needs 12.8)" % level)
PYEOF
need 2_device_smoke "$VENV_VLLM/bin/python" -u "$TOOLS/device_smoke.py" --settings-id "$SETTINGS_ID" --model-dir "$MODEL" --jobs "$JOBS_C4" --call "$CALL" --windows 4 \
    --expect-capability 7.5 --require-no-empty-text --out "$OUT/device-smoke.json"
need 3_windows_capture "$VENV_VLLM/bin/python" -u "$TOOLS/vllm_windows.py" --settings-id "$SETTINGS_ID" --jobs "$JOBS_DEV" --windows "$WINDOWS" --passes batch --capture "$OUT/capture.json" --out "$OUT/windows.json"
need 4_worker_call064 run_worker "$VENV_VLLM/bin/python" vllm "$RUN_C4" "$JOBS_C4" --config "$CONFIG"
need 5_worker_call103 run_worker "$VENV_VLLM/bin/python" vllm "$RUN_103" "$JOBS_103" --config "$CONFIG"
echo "fp16 smoke results in $OUT and $BASE/runs/$RUN_C4 and $BASE/runs/$RUN_103"
