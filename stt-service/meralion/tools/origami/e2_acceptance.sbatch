#!/bin/bash -l
# Job E2 of Stage 3 follow-up E: ONE allocation, one physical GPU, for the ONE configuration that E1 promoted. Structural evidence first (stop before the expensive part when it fails), then the
# Transformers baseline (ALWAYS rerun here, same GPU and commit), then the promoted candidate pilot TWICE (two fresh engines) and its CALL_064 standalone; the historical stock control last (optional).
# Submit ONLY through submit.sh and ONLY for the arm that tools/e1_selection.py promoted:
#   bash $CODE/tools/origami/submit.sh $CODE/tools/origami/e2_acceptance.sbatch l40s 00:42:00 PROMOTED_SETTINGS=<settings id> PROMOTED_CONFIG=<config file name under configs/runtime>
# A default-profile plugin probe must run in the DIAGNOSTIC in-process mode, so that mode is set on its own command line below and nowhere else; a scheduling profile sets its own variable in the decoder.
#SBATCH --job-name=w0-e2
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
: "${PROMOTED_SETTINGS:?PROMOTED_SETTINGS is passed by submit.sh (KEY=VALUE)}"
: "${PROMOTED_CONFIG:?PROMOTED_CONFIG is passed by submit.sh (KEY=VALUE)}"
export CUDA_HOME=${EBROOTCUDA:?CUDA module not loaded}
use_vllm_env
require_clean_commit
cd "$CODE"
OUT=$BASE/e2-$GPU_TAG-$SLURM_JOB_ID
mkdir -p "$OUT"
record_device "$OUT/device.txt"
cat "$OUT/device.txt"
JOBS_PILOT=$BASE/jobs/jobs-pilot22.json
JOBS_C4=$BASE/jobs/jobs-c4.json
WINDOWS=$CODE/tools/origami/problem_windows.json
TOOLS=$CODE/tools/origami
RUNTIME=$CODE/configs/runtime
CANDIDATE_CONFIG=$RUNTIME/$PROMOTED_CONFIG
CONTROL_CONFIG=$RUNTIME/fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-stock.json
TF_PILOT="tf-$GPU_TAG-e2-pilot22-1"
TF_C4="tf-$GPU_TAG-e2-c4-1"
CAND_R1="vllm-$GPU_TAG-e2-cand-pilot22-r1"
CAND_R2="vllm-$GPU_TAG-e2-cand-pilot22-r2"
CAND_C4="vllm-$GPU_TAG-e2-cand-c4-1"
CONTROL="vllm-$GPU_TAG-e2-stockctl-pilot22-1"
test -f "$JOBS_PILOT" && test -f "$JOBS_C4" && test -f "$WINDOWS" && test -f "$CANDIDATE_CONFIG" && test -f "$CONTROL_CONFIG"
for inv in "$TF_PILOT" "$TF_C4" "$CAND_R1" "$CAND_R2" "$CAND_C4" "$CONTROL"; do
  if [ -e "$BASE/runs/$inv/wall.json" ]; then echo "run $inv already finished: refusing to reuse its id"; exit 3; fi
done
# the config must name the promoted settings id; the kind of plugin probe follows from the frozen table (a scheduling profile has an execution entry)
KIND=$("$VENV_VLLM/bin/python" - "$CANDIDATE_CONFIG" "$PROMOTED_SETTINGS" <<'PYEOF'
import sys
from meralion_worker import config as cfg
from meralion_worker.decoders import frozen
path, wanted = sys.argv[1:3]
found = cfg.load_runtime_config(path, env={}).data["decoder"]["settings_id"]
if found != wanted:
    raise SystemExit(f"the config names settings {found!r}, not the promoted {wanted!r}")
print("profile" if frozen.execution_of(wanted) else "diagnostic")
PYEOF
)
echo "promoted settings $PROMOTED_SETTINGS, probe kind: $KIND"
# the declared experiment, written BEFORE any run
"$VENV_VLLM/bin/python" - "$OUT/e2-manifest.json" "$SLURM_JOB_ID" "$MERALION_WORKER_GIT_COMMIT" "$PROMOTED_SETTINGS" "$PROMOTED_CONFIG" "$TF_PILOT" "$TF_C4" "$CAND_R1" "$CAND_R2" "$CAND_C4" "$CONTROL" <<'PYEOF'
import json, re, subprocess, sys
out, job, commit, settings, config, tf_pilot, tf_c4, r1, r2, c4, control = sys.argv[1:12]
smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
uuid = re.search(r"UUID: ((?:GPU|MIG)-[0-9a-f-]+)", smi)
manifest = {"schema": "e2-manifest-v1", "slurm_job_id": job, "worker_commit": commit, "gpu_uuid": uuid.group(1) if uuid else None, "promoted_settings_id": settings, "promoted_config": config,
            "baseline_invocation_ids": {"pilot": tf_pilot, "c4": tf_c4}, "candidate_invocation_ids": [r1, r2], "candidate_c4_invocation_id": c4, "control_invocation_id": control}
open(out, "w").write(json.dumps(manifest, indent=1) + "\n")
print(json.dumps(manifest))
PYEOF
echo "== E2 declared: $OUT/e2-manifest.json"
seconds_left() {
  local left
  left=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null || true)
  python3 - "$left" <<'PYEOF'
import sys
text = sys.argv[1].strip()
if not text:
    print(99999)
else:
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    print(days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2])
PYEOF
}
# a required step: its exit code decides, and nothing after a failed required step runs
need() {
  local name=$1
  shift
  echo "=== step $name"
  set +e
  "$@"
  local rc=$?
  set -e
  echo "=== step $name exit $rc"
  echo "$name $rc" >> "$OUT/steps.txt"
  if [ "$rc" != "0" ]; then echo "required step $name failed (exit $rc): nothing after it runs"; exit "$rc"; fi
}
need 1_controlled_logits_test "$VENV_VLLM/bin/python" "$TOOLS/plugin_logits_test.py" --out "$OUT/plugin_logits_test.json"
need 2_probe_stock_default env VLLM_ENABLE_V1_MULTIPROCESSING=0 "$VENV_VLLM/bin/python" "$TOOLS/vllm_plugin_probe.py" --settings-id stock-vllm-package-v1 --jobs "$JOBS_PILOT" --call CALL_134 --start 24960000 --end 25440000 --out "$OUT/probe-stock.json"
if [ "$PROMOTED_SETTINGS" = "stock-vllm-package-v1" ]; then
  cp "$OUT/probe-stock.json" "$OUT/probe-promoted.json"
elif [ "$KIND" = "profile" ]; then
  need 3_probe_promoted_profile "$VENV_VLLM/bin/python" "$TOOLS/vllm_plugin_probe.py" --settings-id "$PROMOTED_SETTINGS" --jobs "$JOBS_PILOT" --call CALL_134 --start 24960000 --end 25440000 --out "$OUT/probe-promoted.json"
else
  need 3_probe_promoted_diagnostic env VLLM_ENABLE_V1_MULTIPROCESSING=0 "$VENV_VLLM/bin/python" "$TOOLS/vllm_plugin_probe.py" --settings-id "$PROMOTED_SETTINGS" --jobs "$JOBS_PILOT" --call CALL_134 --start 24960000 --end 25440000 --out "$OUT/probe-promoted.json"
fi
need 4_capture_promoted "$VENV_VLLM/bin/python" "$TOOLS/vllm_windows.py" --settings-id "$PROMOTED_SETTINGS" --jobs "$JOBS_PILOT" --windows "$WINDOWS" --passes batch --capture "$OUT/capture.json" --out "$OUT/windows-capture.json"
need 5_tf_pilot run_worker "$VENV_TF/bin/python" transformers "$TF_PILOT" "$JOBS_PILOT"
need 6_tf_c4 run_worker "$VENV_TF/bin/python" transformers "$TF_C4" "$JOBS_C4"
need 7_candidate_pilot_r1 run_worker "$VENV_VLLM/bin/python" vllm "$CAND_R1" "$JOBS_PILOT" --config "$CANDIDATE_CONFIG"
need 8_candidate_c4 run_worker "$VENV_VLLM/bin/python" vllm "$CAND_C4" "$JOBS_C4" --config "$CANDIDATE_CONFIG"
need 9_candidate_pilot_r2 run_worker "$VENV_VLLM/bin/python" vllm "$CAND_R2" "$JOBS_PILOT" --config "$CANDIDATE_CONFIG"
# optional, last, never part of the verdict: the stock default vLLM pilot in the original order (does the historical runaway reproduce in this context?)
LEFT=$(seconds_left)
if [ "$LEFT" -ge 180 ]; then
  set +e
  run_worker "$VENV_VLLM/bin/python" vllm "$CONTROL" "$JOBS_PILOT" --config "$CONTROL_CONFIG"
  echo "10_stock_control_pilot $?" >> "$OUT/steps.txt"
  set -e
else
  echo "10_stock_control_pilot skipped (only $LEFT s left)" >> "$OUT/steps.txt"
fi
echo "E2 results in $OUT and $BASE/runs/*-e2-*"
