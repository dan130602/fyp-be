#!/bin/bash -l
# J3: the vLLM worker on the 22-call pilot (decoded as one batch per recording, the intended pattern). The run id is fixed and never reused. Same FlashInfer rule as J2.
#   cd $BASE/logs && sbatch $CODE/tools/origami/worker_vllm.sbatch
#SBATCH --job-name=w0-vllm
#SBATCH --partition=student
#SBATCH --account=student
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_pro_6000_blackwell_max
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
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
test -f "$JOBS/jobs-pilot22.json"
if [ -e "$BASE/runs/vllm-pilot22-1/wall.json" ]; then echo "run vllm-pilot22-1 already finished: refusing to reuse its id"; exit 3; fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
start_gpu_sampler "$BASE/gpu-vllm-$SLURM_JOB_ID.csv"
trap stop_gpu_sampler EXIT
run_worker "$VENV_VLLM/bin/python" vllm vllm-pilot22-1 "$JOBS/jobs-pilot22.json"
