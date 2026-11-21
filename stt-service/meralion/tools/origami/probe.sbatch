#!/bin/bash -l
# J0: environment probe (about 5 minutes). Records the GPU, driver and CUDA, then checks the model and the installed package against their manifests. No decoding.
#   mkdir -p $BASE/logs && cd $BASE/logs && sbatch $CODE/tools/origami/probe.sbatch
#SBATCH --job-name=w0-probe
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
set -u
source /common/scratch/users/e/YOUR_USERNAME/meralion/runs/worker-v0/code/tools/origami/common.sh
OUT=$BASE/probe-$SLURM_JOB_ID
mkdir -p "$OUT"
{
  nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  echo "apptainer: $(command -v apptainer singularity || echo none)"
  free -g | head -2
  echo "slurm limits: cpus=$SLURM_CPUS_ON_NODE mem_per_node=${SLURM_MEM_PER_NODE:-unset}"
} > "$OUT/system.txt" 2>&1
cat "$OUT/system.txt"
record_device "$OUT/device.txt"
cat "$OUT/device.txt"
if grep -q "MIG" "$OUT/device.txt"; then echo "ALLOCATION: a MIG slice"; else echo "ALLOCATION: a whole GPU (no MIG)"; fi
DEV=$(allocated_device)
ROWS=$(nvidia-smi ${DEV:+-i "$DEV"} --query-gpu=name --format=csv,noheader | grep -c . || true)
echo "sampler rows for the allocated device: $ROWS"
test "$ROWS" = "1" || { echo "the sampler does not select exactly one GPU"; exit 5; }
cd "$CODE"
PY=$VENV_TF/bin/python
$PY -m meralion_worker check-model --model-dir "$MODEL" | tee "$OUT/check-model.json"
$PY -m meralion_worker verify-environment --model-dir "$MODEL" | tee "$OUT/verify-environment.json"
$PY - <<'EOF' | tee "$OUT/imports.txt"
import importlib, inspect, sys
for name in ("meralion_worker.decoders.base", "meralion_worker.decoders.frozen", "meralion_worker.decoders.transformers_decoder", "meralion_worker.decoders.vllm_decoder", "meralion_worker.pipeline"):
    importlib.import_module(name)
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "bf16", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__)
from meralion_3_asr.backends.transformers_backend import TransformersBackend
print("TransformersBackend.__init__", inspect.signature(TransformersBackend.__init__))
print("modules loaded by the worker modules alone: vllm" , "vllm" in sys.modules)
EOF
# vLLM's own device mapping on this allocation (the failure of the first trial): the platform must resolve the allocated device and its capability
VLLM_RC=0
if [ -x "$VENV_VLLM/bin/python" ]; then
  "$VENV_VLLM/bin/python" - <<'PYEOF' 2>&1 | tee "$OUT/vllm-platform.txt" || VLLM_RC=$?
import os, torch
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "bf16", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
print("torch device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0), "total GiB", round(torch.cuda.mem_get_info()[1] / 2**30, 1))
from vllm.platforms import current_platform
print("vllm platform", type(current_platform).__name__, "capability", current_platform.get_device_capability(), "name", current_platform.get_device_name(0))
PYEOF
fi
echo "vllm platform check exit: $VLLM_RC"
echo "probe done: $OUT"
exit $VLLM_RC
