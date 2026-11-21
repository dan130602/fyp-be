#!/bin/bash -l
# Job D2 of Stage 3 follow-up D: ONE allocation, one physical GPU, two separate pinned environments, run back to back: the Transformers baseline (pilot and CALL_064 standalone) and the vLLM
# CANDIDATE (pilot and CALL_064 standalone). The Transformers baseline is ALWAYS rerun here (same GPU, same commit); the earlier L40S baseline recorded no commit.
# Each run starts its own GPU sampler (one file per backend run) and writes run-meta.json. "Same environment" means these agreed comparison conditions, not identical Torch dependencies.
# Submit ONLY through submit.sh, and ONLY after D1 passed:
#   bash $CODE/tools/origami/submit.sh $CODE/tools/origami/compare_candidate.sbatch l40s 00:40:00
#SBATCH --job-name=w0-d2
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
JOBS=$BASE/jobs
CANDIDATE_CONFIG=$CODE/configs/runtime/fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105.json
TF_PILOT="tf-$GPU_TAG-r2-pilot22-1"
TF_C4="tf-$GPU_TAG-r2-c4-1"
VL_PILOT="vllm-$GPU_TAG-rp105-pilot22-1"
VL_C4="vllm-$GPU_TAG-rp105-c4-1"
test -f "$JOBS/jobs-pilot22.json" && test -f "$JOBS/jobs-c4.json" && test -f "$CANDIDATE_CONFIG"
for inv in "$TF_PILOT" "$TF_C4" "$VL_PILOT" "$VL_C4"; do
  if [ -e "$BASE/runs/$inv/wall.json" ]; then echo "run $inv already finished: refusing to reuse its id"; exit 3; fi
done
mkdir -p "$BASE/d2-$GPU_TAG-$SLURM_JOB_ID"
record_device "$BASE/d2-$GPU_TAG-$SLURM_JOB_ID/device.txt"
cat "$BASE/d2-$GPU_TAG-$SLURM_JOB_ID/device.txt"
run_worker "$VENV_TF/bin/python" transformers "$TF_PILOT" "$JOBS/jobs-pilot22.json" || { rc=$?; echo "the Transformers pilot failed (exit $rc): nothing after it runs"; exit $rc; }
run_worker "$VENV_TF/bin/python" transformers "$TF_C4" "$JOBS/jobs-c4.json" || { rc=$?; echo "the Transformers C4 run failed (exit $rc): nothing after it runs"; exit $rc; }
run_worker "$VENV_VLLM/bin/python" vllm "$VL_PILOT" "$JOBS/jobs-pilot22.json" --config "$CANDIDATE_CONFIG" || { rc=$?; echo "the candidate pilot failed (exit $rc): nothing after it runs"; exit $rc; }
run_worker "$VENV_VLLM/bin/python" vllm "$VL_C4" "$JOBS/jobs-c4.json" --config "$CANDIDATE_CONFIG"
