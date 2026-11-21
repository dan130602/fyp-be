#!/bin/bash -l
# One SFT experiment on ONE GPU, phased so the trainer and the vLLM engine are never resident together:
#   (optional) E00_FIRST=1: the matching 35-call E00 evaluation of the untouched base weights, written once to $BASELINE
#   for each epoch:  trainer process (resume) -> exits after the epoch (LoRA checkpoint, teacher-forced validation, logs and resumable state written, THEN the completion marker)
#                    -> eval_ckpt.sh: temporary merged export -> unchanged worker (vLLM BF16, vllm-rp105-sched0-v1) on the 35 validation calls -> scorer -> metrics.json
# An epoch is finished ONLY when checkpoints/epoch-N.complete.json exists (the .safetensors file alone proves nothing: it is written before validation and the resumable state).
# Submit (approved: E01 and E02 only, K=1067, 3 epochs, 04:00:00 each; E01 also runs E00 first):
#   bash submit_sft.sh train_job.sbatch blackwell 04:00:00 RUN_ID=E01 OBJECTIVE=ce  SAMPLING=uniform K=1067 E00_FIRST=1 BASELINE=/common/scratch/users/e/YOUR_USERNAME/meralion/sft/eval/E00-35call
#   bash submit_sft.sh train_job.sbatch blackwell 04:00:00 RUN_ID=E02 OBJECTIVE=wce SAMPLING=uniform K=1067            BASELINE=/common/scratch/users/e/YOUR_USERNAME/meralion/sft/eval/E00-35call
# Optional LR=<value>. DEFAULT 2.5e-5 (the best of the 5e-5 / 2.5e-5 / 1e-5 sweep; the first E01/E02 used 5e-5, so pass LR=5e-5 to reproduce them). Always use a NEW RUN_ID.
# Optional ENTITY_WEIGHT=<w> (default 2; only used when OBJECTIVE=wce): the weighted run's entity-token weight; name such runs by weight, e.g. RUN_ID=E02-W4.
# Teacher-forced validation runs at every epoch end AND mid-epoch (trainer default --validate-per-epoch 2); step-*.safetensors are the mid-epoch adapters.
# Optional TARGET_POOL=entity|important (default entity: the targeted extra draws come from the clips that contain entity spans; important = the original broader rule used by the first E03).
# Optional SEED=<n> (default 42): LoRA init AND the sampling plan; the seed-replicate runs use a different value, e.g. RUN_ID=E01-LR25-seed43 SEED=43.
# Optional FEATURE_CACHE=<dir> from sft_features.py (verified bit-identical processor output; skips recomputing it every step).
# Optional SCOPE=decoder|adapter-lora|adapter-full|encoder-lora (default decoder = E01-E04): what is trained, everything else frozen. adapter-full trains every speech-adapter parameter (LR applies to the FP32 weights directly);
#   the other scopes are LoRA (rank 8, alpha 16) on the speech adapter's Linear layers / the encoder's attention + MLP projections. Name such runs by scope, e.g. RUN_ID=E06-LORA SCOPE=adapter-lora.
# Optional BASE_MODEL=<model dir> and BASE_MANIFEST=<model manifest json>: train from (and export against) another model directory, e.g. the 3x adapter model built by sft_arch3.py (A02). Default: the pinned 5x model.
# Optional AUGMENT=none|extras: extras = the K extra draws per epoch become augmented copies (synthetic noise 0-20 dB SNR, telephone, muffled); every clip is still seen clean once per epoch. Same budget and time.
# Optional EPOCH0_REFERENCE=<epoch_summary.jsonl of E01-LR25>: the run stops before training unless its epoch-0 validation (untouched model) reproduces that run's epoch 0 within 1e-3 nats (trainer --epoch0-tolerance).
# Resubmitting with the same RUN_ID continues from the saved state (--resume) and skips finished epochs.
#SBATCH --job-name=sft-train
#SBATCH --partition=student
#SBATCH --account=student
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
set -eo pipefail
module load Python/3.12.8-GCCcore-13.3.0
module load CUDA/12.8.0
set -u
: "${RUN_ID:?RUN_ID}"; : "${OBJECTIVE:?ce|wce}"; : "${SAMPLING:?uniform|targeted}"; : "${K:?K}"
EPOCHS=${EPOCHS:-3}; EVAL=${EVAL:-1}; BASELINE=${BASELINE:-}; E00_FIRST=${E00_FIRST:-0}; DATA=${DATA:-data-v2}; LR=${LR:-2.5e-5}; FEATURE_CACHE=${FEATURE_CACHE:-}; ENTITY_WEIGHT=${ENTITY_WEIGHT:-2}; SEED=${SEED:-42}; TARGET_POOL=${TARGET_POOL:-entity}; SCOPE=${SCOPE:-decoder}; AUGMENT=${AUGMENT:-none}; EPOCH0_REFERENCE=${EPOCH0_REFERENCE:-}
SFT=/common/scratch/users/e/YOUR_USERNAME/meralion/sft
PILOT=$HOME/meralion-pilot
WORKER=/common/scratch/users/e/YOUR_USERNAME/meralion/runs/worker-v0
source $WORKER/code/tools/origami/common.sh
BASE_MODEL=${BASE_MODEL:-$MODEL}; export BASE_MODEL BASE_MANIFEST       # A02 trains from, and exports against, the 3x model directory
RUNS=$SFT/runs
RUN=$RUNS/$RUN_ID
mkdir -p "$RUN/logs" "$RUN/eval"
record_device "$RUN/logs/device-$SLURM_JOB_ID.txt"; cat "$RUN/logs/device-$SLURM_JOB_ID.txt"
echo "job $SLURM_JOB_ID run $RUN_ID objective $OBJECTIVE sampling $SAMPLING K $K epochs $EPOCHS lr $LR entity-weight $ENTITY_WEIGHT seed $SEED target-pool $TARGET_POOL scope $SCOPE feature-cache '$FEATURE_CACHE' data $DATA baseline '$BASELINE' started $(date -Iseconds)"
TRAIN_PY=$PILOT/.venv-cu128/bin/python
export PYTHONPATH=$SFT/code:$PILOT/finetune-deps-v1 PYTHONUNBUFFERED=1

FILES=$(find /common/scratch/users/e/YOUR_USERNAME 2>/dev/null | wc -l)
echo "scratch file count $FILES (quota 100000)"
[ "$FILES" -lt 97500 ] || { echo "refusing to start: fewer than 2,500 files of quota left"; exit 5; }
if [ "$E00_FIRST" = "1" ] && [ -n "$BASELINE" ] && [ ! -f "$BASELINE/metrics.json" ]; then
  echo "=== E00 (35 calls, base weights) $(date +%T)"
  bash $SFT/code/eval_ckpt.sh E00 base base "$BASELINE" 2>&1 | tee -a "$RUN/logs/e00-$SLURM_JOB_ID.log"
fi

trap 'stop_gpu_sampler' EXIT
for ((e = 1; e <= EPOCHS; e++)); do
  if [ -f "$RUN/checkpoints/epoch-$e.complete.json" ]; then echo "epoch $e already complete"; else
    RESUME=""; [ -f "$RUN/checkpoints/latest-state.pt" ] && RESUME="--resume"
    start_gpu_sampler "$RUN/logs/gpu-$SLURM_JOB_ID-train-epoch-$e.csv"
    $TRAIN_PY -u $SFT/code/sft_train.py --run-id $RUN_ID --out-root $RUNS --data $SFT/$DATA --model $BASE_MODEL --batch $PILOT/batch \
        --objective $OBJECTIVE --sampling $SAMPLING --k $K --epochs $EPOCHS --lr $LR --entity-weight $ENTITY_WEIGHT --seed $SEED --target-pool $TARGET_POOL --scope $SCOPE --augment $AUGMENT ${EPOCH0_REFERENCE:+--epoch0-reference $EPOCH0_REFERENCE} ${FEATURE_CACHE:+--feature-cache $FEATURE_CACHE} --stop-after-epoch $e $RESUME 2>&1 | tee -a "$RUN/logs/train-$SLURM_JOB_ID.log"
    stop_gpu_sampler
    [ -f "$RUN/checkpoints/epoch-$e.complete.json" ] || { echo "epoch $e did not complete (no completion marker): stopping"; exit 2; }
  fi
  if [ "$EVAL" = "1" ] && [ ! -f "$RUN/eval/epoch-$e/metrics.json" ]; then
    rm -rf -- "$RUN/eval/epoch-$e"            # an interrupted evaluation leaves partial files: start it clean (exact path under this run)
    bash $SFT/code/eval_ckpt.sh $RUN_ID epoch-$e "$RUN/checkpoints/epoch-$e.safetensors" "$RUN/eval/epoch-$e" $BASELINE 2>&1 | tee -a "$RUN/logs/eval-epoch-$e-$SLURM_JOB_ID.log"
    [ -f "$RUN/eval/epoch-$e/metrics.json" ] || { echo "evaluation of epoch $e failed: stopping"; exit 3; }
  fi
done
echo "RUN_DONE $RUN_ID $(date -Iseconds)"
