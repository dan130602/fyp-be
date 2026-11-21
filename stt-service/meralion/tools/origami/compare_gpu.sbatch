#!/bin/bash -l
# The same-GPU comparison (Stage 3 follow-up, C): ONE allocation, one physical GPU, two separate pinned environments, run back to back:
#   Transformers pilot22, Transformers CALL_064 alone, vLLM pilot22, vLLM CALL_064 alone (the standalone references are kept for each backend).
# Submit ONLY through submit.sh, which passes the GPU type and the time limit on the sbatch command line and checks the GPU-time ledger:
#   bash $CODE/tools/origami/submit.sh $CODE/tools/origami/compare_gpu.sbatch <l40s|a40|3090|blackwell> <HH:MM:SS>
# The directive defaults below are deliberately small, so a direct sbatch cannot spend real GPU time. Run ids carry the GPU tag and are never reused.
#SBATCH --job-name=w0-compare
#SBATCH --partition=student
#SBATCH --account=student
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_pro_6000_blackwell_max
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log

set -eo pipefail
module load Python/3.12.8-GCCcore-13.3.0
module load CUDA/12.8.0
set -u
source /common/scratch/users/e/YOUR_USERNAME/meralion/runs/worker-v0/code/tools/origami/common.sh
: "${GPU_TAG:?GPU_TAG is set by submit.sh}"
export CUDA_HOME=${EBROOTCUDA:?CUDA module not loaded}
use_vllm_env
cd "$CODE"
JOBS=$BASE/jobs
test -f "$JOBS/jobs-pilot22.json" && test -f "$JOBS/jobs-c4.json"
for inv in "tf-$GPU_TAG-pilot22-1" "tf-$GPU_TAG-c4-1" "vllm-$GPU_TAG-pilot22-1" "vllm-$GPU_TAG-c4-1"; do
  if [ -e "$BASE/runs/$inv/wall.json" ]; then echo "run $inv already finished: refusing to reuse its id"; exit 3; fi
done
mkdir -p "$BASE/compare-$GPU_TAG-$SLURM_JOB_ID"
record_device "$BASE/compare-$GPU_TAG-$SLURM_JOB_ID/device.txt"
cat "$BASE/compare-$GPU_TAG-$SLURM_JOB_ID/device.txt"
start_gpu_sampler "$BASE/gpu-compare-$GPU_TAG-$SLURM_JOB_ID.csv"
trap stop_gpu_sampler EXIT
run_worker "$VENV_TF/bin/python" transformers "tf-$GPU_TAG-pilot22-1" "$JOBS/jobs-pilot22.json" || { rc=$?; echo "the Transformers pilot failed (exit $rc): nothing after it runs"; exit $rc; }
run_worker "$VENV_TF/bin/python" transformers "tf-$GPU_TAG-c4-1" "$JOBS/jobs-c4.json" || { rc=$?; echo "the Transformers C4 run failed (exit $rc): nothing after it runs"; exit $rc; }
run_worker "$VENV_VLLM/bin/python" vllm "vllm-$GPU_TAG-pilot22-1" "$JOBS/jobs-pilot22.json" || { rc=$?; echo "the vLLM pilot failed (exit $rc): nothing after it runs"; exit $rc; }
run_worker "$VENV_VLLM/bin/python" vllm "vllm-$GPU_TAG-c4-1" "$JOBS/jobs-c4.json"
