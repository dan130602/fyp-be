#!/bin/bash -l
# J2: the vLLM adapter smoke (up to 45 minutes). (i) live D1/D6/D7 checks on the first 4 windows of the shortest pilot call, (ii) the one-recording C4 invocation with the vLLM backend.
# A FlashInfer, install or import failure ENDS the vLLM trial here: no other attention backend is tried (the frozen constructor defaults are part of the comparison).
#   bash $CODE/tools/origami/submit.sh $CODE/tools/origami/vllm_smoke.sbatch <l40s|a40|3090|blackwell> <HH:MM:SS>
#SBATCH --job-name=w0-vllm-smoke
#SBATCH --partition=student
#SBATCH --account=student
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_pro_6000_blackwell_max
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.log

set -eo pipefail
module load Python/3.12.8-GCCcore-13.3.0
module load CUDA/12.8.0
set -u
source /common/scratch/users/e/YOUR_USERNAME/meralion/runs/worker-v0/code/tools/origami/common.sh
export CUDA_HOME=${EBROOTCUDA:?CUDA module not loaded}
use_vllm_env
cd "$CODE"
JOBS=$BASE/jobs
test -f "$JOBS/jobs-c4.json"
: "${GPU_TAG:?GPU_TAG is set by submit.sh}"
RUN_ID="vllm-smoke-$GPU_TAG-c4-1"
if [ -e "$BASE/runs/$RUN_ID/wall.json" ]; then echo "run $RUN_ID already finished: refusing to reuse its id"; exit 3; fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
mkdir -p "$BASE/smoke"
record_device "$BASE/smoke/device-$SLURM_JOB_ID.txt"
start_gpu_sampler "$BASE/gpu-vllm-smoke-$SLURM_JOB_ID.csv"
trap stop_gpu_sampler EXIT
CALL=$(python3 -c "import json; print(json.load(open('$JOBS/jobs-c4.json'))['jobs'][0]['job_name'])")
AUDIO=$(python3 -c "import json; print(json.load(open('$JOBS/jobs-c4.json'))['jobs'][0]['input'])")
mkdir -p "$BASE/smoke"
"$VENV_VLLM/bin/python" -u "$CODE/tools/origami/vllm_smoke.py" --model-dir "$MODEL" --audio "$AUDIO" --call "$CALL" --windows 4 --out "$BASE/smoke/vllm-smoke-$SLURM_JOB_ID.json"
run_worker "$VENV_VLLM/bin/python" vllm "$RUN_ID" "$JOBS/jobs-c4.json"
