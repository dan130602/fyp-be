#!/bin/bash -l
# Job D1 of Stage 3 follow-up D: construct the stock decoders, test the no-repeat-ngram processor directly, probe it live (isolated diagnostic processes), then run the problem windows and the candidate's
# repeatability in the NORMAL multiprocess mode. Every step is its OWN process and writes its OWN result file, so a step that fails or times out loses nothing else.
# Submit ONLY through submit.sh (GPU type and time limit on the command line, ledger checked first):
#   bash $CODE/tools/origami/submit.sh $CODE/tools/origami/d1_diagnostics.sbatch l40s 00:20:00
# The VLLM_ENABLE_V1_MULTIPROCESSING=0 diagnostic mode is set ONLY on the two probe command lines below, never exported, never in common.sh.
#SBATCH --job-name=w0-d1
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
OUT=$BASE/d1-$GPU_TAG-$SLURM_JOB_ID
mkdir -p "$OUT"
record_device "$OUT/device.txt"
cat "$OUT/device.txt"
JOBS=$BASE/jobs/jobs-pilot22.json
WINDOWS=$CODE/tools/origami/problem_windows.json
TOOLS=$CODE/tools/origami
CANDIDATE=vllm-rp105-v1
start_gpu_sampler "$OUT/gpu.csv"
trap stop_gpu_sampler EXIT
FAILED=""
step() {
  local name=$1
  shift
  echo "=== step $name"
  set +e
  "$@"
  local rc=$?
  set -e
  echo "=== step $name exit $rc"
  echo "$name $rc" >> "$OUT/steps.txt"
  if [ "$rc" != "0" ]; then FAILED="$FAILED $name"; fi
}
step 1a_construct_transformers_stock "$VENV_TF/bin/python" "$TOOLS/construct_stock_decoders.py" --backend transformers --out "$OUT/construct-transformers.json"
step 1b_construct_vllm_stock "$VENV_VLLM/bin/python" "$TOOLS/construct_stock_decoders.py" --backend vllm --out "$OUT/construct-vllm-stock.json"
step 2_controlled_logits_test "$VENV_VLLM/bin/python" "$TOOLS/plugin_logits_test.py" --out "$OUT/plugin_logits_test.json"
step 3_probe_stock env VLLM_ENABLE_V1_MULTIPROCESSING=0 "$VENV_VLLM/bin/python" "$TOOLS/vllm_plugin_probe.py" --settings-id stock-vllm-package-v1 --jobs "$JOBS" --call CALL_134 --start 24960000 --end 25440000 --out "$OUT/probe-stock.json"
step 4_probe_candidate env VLLM_ENABLE_V1_MULTIPROCESSING=0 "$VENV_VLLM/bin/python" "$TOOLS/vllm_plugin_probe.py" --settings-id $CANDIDATE --jobs "$JOBS" --call CALL_134 --start 24960000 --end 25440000 --out "$OUT/probe-candidate.json"
step 5_windows_stock_normal_mode "$VENV_VLLM/bin/python" "$TOOLS/vllm_windows.py" --settings-id stock-vllm-package-v1 --jobs "$JOBS" --windows "$WINDOWS" --passes batch --package-parity --out "$OUT/windows-stock.json"
step 6_windows_candidate_R1_normal_mode "$VENV_VLLM/bin/python" "$TOOLS/vllm_windows.py" --settings-id $CANDIDATE --jobs "$JOBS" --windows "$WINDOWS" --passes batch,batch,single --package-parity --out "$OUT/windows-candidate-R1.json"
step 7_windows_candidate_R2_fresh_engine "$VENV_VLLM/bin/python" "$TOOLS/vllm_windows.py" --settings-id $CANDIDATE --jobs "$JOBS" --windows "$WINDOWS" --passes batch --out "$OUT/windows-candidate-R2.json"
echo "steps failed:${FAILED:- none}"
echo "D1 results in $OUT"
test -z "$FAILED"
