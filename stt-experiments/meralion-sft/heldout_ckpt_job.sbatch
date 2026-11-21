#!/bin/bash -l
# One SFT checkpoint (LoRA merged into BF16 base weights) on the 75 clean HELD-OUT recordings with the FROZEN inference pipeline (vllm-rp105-sched0-v1 + the runtime config named below), ONE run.
# Inference only: no reference text is on the cluster; scoring happens on the laptop. The held-out set is for REPORTING: declare the candidate list BEFORE running and run every candidate once.
#   bash submit_sft.sh heldout_ckpt_job.sbatch blackwell 00:40:00 RUN_ID=E01-LR25 CKPT=epoch-2 LABEL=heldout75-clean-sft-ce-uniform-ep2
#   noisy set: ... SET_DIR=noisy-heldout75 LABEL=heldout75-noisy-sft-ce-uniform-ep2-20261119 (the same declared candidate, run once on the noisy audio)
# LABEL becomes the output folder $SFT/heldout/<LABEL> and part of the worker invocation id. Fetch: tar the folder's worker/out/transcripts, then stamp_sft_predictions.py + import_predictions.py --allow-adapter.
#SBATCH --job-name=heldout-ckpt
#SBATCH --partition=student
#SBATCH --account=student
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.log
set -eo pipefail
module load Python/3.12.8-GCCcore-13.3.0
module load CUDA/12.8.0
set -u
: "${RUN_ID:?training run id, e.g. E01-LR25}"; : "${CKPT:?checkpoint name, e.g. epoch-2}"; : "${LABEL:?output label, e.g. heldout75-clean-sft-ce-uniform-ep2}"
SFT=/common/scratch/users/e/YOUR_USERNAME/meralion/sft
PILOT=$HOME/meralion-pilot
WORKER=/common/scratch/users/e/YOUR_USERNAME/meralion/runs/worker-v0
source $WORKER/code/tools/origami/common.sh
export CUDA_HOME=${EBROOTCUDA:?CUDA module not loaded}
WCODE=$SFT/worker-code
CONFIG_IN=$WORKER/code/configs/runtime/fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0.json
LORA=$SFT/runs/$RUN_ID/checkpoints/$CKPT.safetensors
SET_DIR=${SET_DIR:-heldout75}                                       # heldout75 (clean) or noisy-heldout75 (the environment_v2 noisy / channel-degraded variant of the same 75 calls)
HELD=/common/scratch/users/e/YOUR_USERNAME/meralion/$SET_DIR
JOBS=$HELD/jobs-$SET_DIR.json
OUT=$SFT/heldout/$LABEL
EXPORT=$SFT/exports/$LABEL-tmp
TRAIN_PY=$PILOT/.venv-cu128/bin/python
test -f "$LORA" && test -f "$JOBS" && test -f "$CONFIG_IN" && test -d "$WCODE"
[ ! -e "$OUT" ] || { echo "refusing to reuse $OUT (a held-out evaluation is run once per label)"; exit 3; }
mkdir -p "$OUT"
record_device "$OUT/device.txt"; cat "$OUT/device.txt"
case "$EXPORT" in $SFT/exports/*-tmp) rm -rf -- "$EXPORT" ;; *) echo "refusing to delete $EXPORT"; exit 3 ;; esac
MID=meralion-3-3b-asr-sft-$LABEL
PYTHONPATH=$SFT/code:$PILOT/finetune-deps-v1 $TRAIN_PY -u $SFT/code/sft_export.py --base $MODEL --lora "$LORA" --out "$EXPORT" --id $MID --base-manifest $WCODE/configs/models/meralion-3-3b-asr-base-9f226d7.json \
    --manifest-out $WCODE/configs/models/$MID.json --runtime-config-in $CONFIG_IN --runtime-config-out $OUT/runtime-config.json | tee "$OUT/export.log"
use_vllm_env
start_gpu_sampler "$OUT/gpu.csv"
set +e
PYTHONPATH=$WCODE timeout 1800 $VENV_VLLM/bin/python -u -m meralion_worker run --invocation-id "$LABEL-$(date +%s)" --backend vllm --jobs "$JOBS" \
    --model-dir "$EXPORT" --config "$OUT/runtime-config.json" --output-dir "$OUT/worker/out" --state-dir "$OUT/worker/state" --summary-output "$OUT/worker/summary.json" 2>&1 | tee "$OUT/worker.log"
RC=${PIPESTATUS[0]}
set -e
stop_gpu_sampler
echo "worker exit $RC"
case "$EXPORT" in $SFT/exports/*-tmp) rm -rf -- "$EXPORT" ;; esac
[ "$RC" = "0" ] || exit $RC
echo "HELDOUT_CKPT_DONE $LABEL $(date -Iseconds)"
