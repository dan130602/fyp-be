#!/bin/bash -l
# J1: the Transformers worker, no Docker. (a) the 22-call pilot invocation, then (b) the one-recording C4 reference (the shortest pilot call). Run ids are fixed and never reused.
#   cd $BASE/logs && sbatch $CODE/tools/origami/worker_tf.sbatch
#SBATCH --job-name=w0-tf
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
set -u
source /common/scratch/users/e/YOUR_USERNAME/meralion/runs/worker-v0/code/tools/origami/common.sh
cd "$CODE"
JOBS=$BASE/jobs
test -f "$JOBS/jobs-pilot22.json" && test -f "$JOBS/jobs-c4.json"
for inv in tf-pilot22-1 tf-c4-1; do
  if [ -e "$BASE/runs/$inv/wall.json" ]; then echo "run $inv already finished: refusing to reuse its id"; exit 3; fi
done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
start_gpu_sampler "$BASE/gpu-tf-$SLURM_JOB_ID.csv"
trap stop_gpu_sampler EXIT
run_worker "$VENV_TF/bin/python" transformers tf-pilot22-1 "$JOBS/jobs-pilot22.json" || { rc=$?; echo "the pilot invocation failed (exit $rc): the C4 reference is not run"; exit $rc; }
run_worker "$VENV_TF/bin/python" transformers tf-c4-1 "$JOBS/jobs-c4.json"
