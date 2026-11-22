#!/bin/bash -l
# Job E1 of Stage 3 follow-up E: four vLLM configurations (arms), TWO fresh engines each, on the three complete problem recordings CALL_134, CALL_103 and CALL_217 in that fixed order, plus one
# 9-window process under the deterministic-scheduling candidate profile. Every worker run is its own process (own sampler, run-meta, documents, state); a failed run never discards another's files.
#   A  vllm-rp105-v1               candidate, default scheduling      B  vllm-rp105-sched0-v1        candidate, deterministic scheduling
#   C  stock-vllm-package-v1       stock, default scheduling          D  stock-vllm-package-sched0-v1  stock, deterministic scheduling
# The scheduling profile is a settings id (a scheduling profile sets its own environment variable inside the decoder): nothing in this script sets that variable.
# Submit ONLY through submit.sh (GPU type and time limit on the command line, ledger checked first):
#   bash $CODE/tools/origami/submit.sh $CODE/tools/origami/e1_profiles.sbatch l40s 00:15:00
#SBATCH --job-name=w0-e1
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
OUT=$BASE/e1-$GPU_TAG-$SLURM_JOB_ID
mkdir -p "$OUT"
record_device "$OUT/device.txt"
cat "$OUT/device.txt"
JOBS_PILOT=$BASE/jobs/jobs-pilot22.json
JOBS3=$BASE/jobs/jobs-e1-3rec.json
WINDOWS=$CODE/tools/origami/problem_windows.json
TOOLS=$CODE/tools/origami
RUNTIME=$CODE/configs/runtime
STEM=fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget
CONFIG_A=$RUNTIME/$STEM-vllm-rp105.json
CONFIG_B=$RUNTIME/$STEM-vllm-rp105-sched0.json
CONFIG_C=$RUNTIME/$STEM-vllm-stock.json
CONFIG_D=$RUNTIME/$STEM-vllm-stock-sched0.json
test -f "$JOBS_PILOT" && test -f "$WINDOWS" && test -f "$CONFIG_A" && test -f "$CONFIG_B" && test -f "$CONFIG_C" && test -f "$CONFIG_D"
# the three complete recordings, in the fixed order, from the pilot jobs file (same audio paths); an existing file must already be identical
"$VENV_VLLM/bin/python" - "$JOBS_PILOT" "$JOBS3" <<'PYEOF'
import json, sys
pilot, target = sys.argv[1:3]
wanted = ["CALL_134", "CALL_103", "CALL_217"]
by_name = {job["job_name"]: job for job in json.load(open(pilot))["jobs"]}
missing = [name for name in wanted if name not in by_name]
if missing:
    raise SystemExit(f"jobs missing from the pilot jobs file: {missing}")
document = {"jobs": [by_name[name] for name in wanted]}
try:
    existing = json.load(open(target))
except FileNotFoundError:
    open(target, "w").write(json.dumps(document, indent=1) + "\n")
else:
    if existing != document:
        raise SystemExit(f"{target} exists and differs from the expected three-recording jobs file")
PYEOF
declare -A CONFIG=([A]="$CONFIG_A" [B]="$CONFIG_B" [C]="$CONFIG_C" [D]="$CONFIG_D")
for arm in A B C D; do
  for n in 1 2; do
    if [ -e "$BASE/runs/vllm-$GPU_TAG-e1-$arm-$n/wall.json" ]; then echo "run vllm-$GPU_TAG-e1-$arm-$n already finished: refusing to reuse its id"; exit 3; fi
  done
done
# the declared experiment, written BEFORE any run
"$VENV_VLLM/bin/python" - "$OUT/e1-manifest.json" "$GPU_TAG" "$SLURM_JOB_ID" "$MERALION_WORKER_GIT_COMMIT" "$CONFIG_A" "$CONFIG_B" "$CONFIG_C" "$CONFIG_D" <<'PYEOF'
import json, re, subprocess, sys
from meralion_worker import config as cfg
out, tag, job, commit = sys.argv[1:5]
configs = dict(zip("ABCD", sys.argv[5:9]))
smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
uuid = re.search(r"UUID: ((?:GPU|MIG)-[0-9a-f-]+)", smi)
arms = {arm: {"config": path, "settings_id": cfg.load_runtime_config(path, env={}).data["decoder"]["settings_id"], "run_ids": [f"vllm-{tag}-e1-{arm}-{n}" for n in (1, 2)]} for arm, path in configs.items()}
manifest = {"schema": "e1-manifest-v1", "jobs": ["CALL_134", "CALL_103", "CALL_217"], "slurm_job_id": job, "worker_commit": commit, "gpu_uuid": uuid.group(1) if uuid else None, "arms": arms,
            "order": ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"]}
open(out, "w").write(json.dumps(manifest, indent=1) + "\n")
print(json.dumps(manifest))
PYEOF
FAILED=""
step() {
  local name=$1
  shift
  echo "=== step $name"
  set +e
  "$@"
  local rc=$?
  set -e
  echo "=== step $name exit $rc"
  echo "$name $rc" >> "$OUT/steps.txt"
  if [ "$rc" != "0" ]; then FAILED="$FAILED $name"; fi
}
for n in 1 2; do
  for arm in A B C D; do
    step "${arm}${n}_worker_run" run_worker "$VENV_VLLM/bin/python" vllm "vllm-$GPU_TAG-e1-$arm-$n" "$JOBS3" --config "${CONFIG[$arm]}"
  done
done
step 9_windows_B_profile "$VENV_VLLM/bin/python" "$TOOLS/vllm_windows.py" --settings-id vllm-rp105-sched0-v1 --jobs "$JOBS_PILOT" --windows "$WINDOWS" --passes batch,batch,single --out "$OUT/windows-B-R1.json"
echo "steps failed:${FAILED:- none}"
echo "E1 results in $OUT and $BASE/runs/vllm-$GPU_TAG-e1-*"
test -z "$FAILED"
