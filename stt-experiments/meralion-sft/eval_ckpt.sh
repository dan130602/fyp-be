#!/bin/bash
# Evaluate ONE checkpoint (or the base model) with the frozen inference pipeline and score it. Runs INSIDE a GPU allocation, after the trainer process has exited (the trainer is never loaded at the same time).
#   bash eval_ckpt.sh <EXP> <CKPT_ID> <lora .safetensors | base> <OUT_DIR> [BASELINE_EVAL_DIR]
# Frozen: BF16 vLLM profile vllm-rp105-sched0-v1, the fg175 ... -rp105-sched0 runtime config (FG 30 s/15 s, M2-AX, retries as capped there, percentages-v2 + amount-v1), unchanged worker code and scorer.
# One ASR pass per checkpoint; raw / percentage / final scores all come from it. The merged export is temporary and removed afterwards (exact path under $SFT/exports only).
set -eo pipefail
EXP=${1:?exp}; CKPT=${2:?ckpt id}; LORA=${3:?lora file or base}; OUT=${4:?out dir}; BASELINE=${5:-}
SFT=/common/scratch/users/e/YOUR_USERNAME/meralion/sft
PILOT=$HOME/meralion-pilot
WORKER=/common/scratch/users/e/YOUR_USERNAME/meralion/runs/worker-v0
source $WORKER/code/tools/origami/common.sh
export CUDA_HOME=${EBROOTCUDA:?CUDA module not loaded}
WCODE=$SFT/worker-code
[ -d "$WCODE" ] || cp -r $WORKER/code "$WCODE"
CONFIG_IN=$WORKER/code/configs/runtime/fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0.json
# the baseline (E00) may still be running in the other job: wait for its metrics (the error CSVs compare against it); never score a checkpoint against a missing baseline
if [ -n "$BASELINE" ] && [ "$EXP" != "E00" ]; then
  waited=0
  until [ -f "$BASELINE/metrics.json" ]; do
    [ $waited -ge 3000 ] && { echo "baseline $BASELINE/metrics.json did not appear within 50 min"; exit 4; }
    sleep 20; waited=$((waited + 20))
  done
fi
if [ -d "$OUT/worker" ] && [ ! -f "$OUT/metrics.json" ]; then rm -rf -- "$OUT/worker"; fi      # leftovers of an interrupted attempt of THIS evaluation (exact path)
mkdir -p "$OUT"
TRAIN_PY=$PILOT/.venv-cu128/bin/python
if [ "$LORA" = "base" ]; then
  MODEL_DIR=${BASE_MODEL:-$MODEL}; CONFIG=$CONFIG_IN; EXPORT=""
  if [ -n "${BASE_MODEL_ID:-}" ]; then      # an UNTRAINED variant model directory (A02's transformed 3x init): the worker must check it against ITS manifest, not the pinned 5x model's
    CONFIG=$OUT/runtime-config.json
    $TRAIN_PY - "$CONFIG_IN" "$CONFIG" "$BASE_MODEL_ID" <<'PYEOF'
import json, sys
src, dst, mid = sys.argv[1:]
cfg = json.load(open(src))
cfg["model_manifest"] = mid
cfg["config_id"] = cfg["config_id"] + "-" + mid
json.dump(cfg, open(dst, "w"), indent=2)
PYEOF
  fi
else
  EXPORT=$SFT/exports/$EXP-tmp
  case "$EXPORT" in $SFT/exports/*-tmp) rm -rf -- "$EXPORT" ;; *) echo "refusing to delete $EXPORT"; exit 3 ;; esac
  MID=meralion-3-3b-asr-sft-$EXP-$CKPT
  PYTHONPATH=$SFT/code:$PILOT/finetune-deps-v1 $TRAIN_PY -u $SFT/code/sft_export.py --base ${BASE_MODEL:-$MODEL} --lora "$LORA" --out "$EXPORT" --id $MID --base-manifest ${BASE_MANIFEST:-$WCODE/configs/models/meralion-3-3b-asr-base-9f226d7.json} \
      --manifest-out $WCODE/configs/models/$MID.json --runtime-config-in $CONFIG_IN --runtime-config-out $OUT/runtime-config.json | tee "$OUT/export.log"
  MODEL_DIR=$EXPORT; CONFIG=$OUT/runtime-config.json
fi
# the 35 validation calls (development calls only), in manifest order
"$VENV_VLLM/bin/python" - "$PILOT/batch" "$SFT/${DATA:-data-v2}/validation_calls.json" "$OUT/jobs.json" <<'PYEOF'
import json, sys
batch, val, out = sys.argv[1:]
ids = json.load(open(val))["call_ids"]
rows = {c["call_id"]: c for c in json.load(open(f"{batch}/calls.json"))["calls"]}
json.dump({"jobs": [{"job_name": i, "input": f"{batch}/{rows[i]['audio']}"} for i in ids]}, open(out, "w"), indent=1)
PYEOF
use_vllm_env
start_gpu_sampler "$OUT/gpu.csv"
set +e
PYTHONPATH=$WCODE timeout ${EVAL_TIMEOUT:-1500} $VENV_VLLM/bin/python -u -m meralion_worker run --invocation-id "$EXP-$CKPT-$(date +%s)" --backend vllm --jobs "$OUT/jobs.json" \
    --model-dir "$MODEL_DIR" --config "$CONFIG" --output-dir "$OUT/worker/out" --state-dir "$OUT/worker/state" --summary-output "$OUT/worker/summary.json" 2>&1 | tee "$OUT/worker.log"
RC=${PIPESTATUS[0]}
set -e
stop_gpu_sampler
echo "worker exit $RC"
if [ -n "$EXPORT" ]; then rm -rf -- "$EXPORT"; fi
[ "$RC" = "0" ] || exit $RC
ARGS=(score --docs "$OUT/worker/out/transcripts" --exp "$EXP" --ckpt "$CKPT" --out "$OUT" --split $SFT/${DATA:-data-v2}/split.json --valid $SFT/${DATA:-data-v2}/validation_calls.json --worker-summary "$OUT/worker/summary.json" --worker-root "$WCODE" --engine-root $SFT/engine/fyp-be/evaluation-engine --tag "-$(date +%m%d%H%M%S)")
[ -n "$BASELINE" ] && ARGS+=(--baseline "$BASELINE")
PYTHONPATH=$SFT/code:$SFT/pydeps:$WCODE $TRAIN_PY -X utf8 -u $SFT/code/sft_eval.py "${ARGS[@]}" 2>&1 | tee "$OUT/score.log"
# the scratch area has a FILE-count quota (100,000): every evaluation except the E00 baseline (later evaluations compare against its engine runs) is packed into tar.gz files
if [ "$EXP" != "E00" ] && [ -f "$OUT/metrics.json" ]; then
  $TRAIN_PY $SFT/code/pack_eval.py --out "$OUT" --engine-runs $SFT/engine/fyp-be/evaluation-engine/runs
fi
