#!/bin/bash -l
# Full development validation of ONE FIXED configuration: settings vllm-rp105-sched0-v1 (the config named below), the whole 175-recording development set, TWICE, each on a fresh engine in its own process,
# in ONE allocation on ONE physical GPU. No Transformers run (the comparison baseline is the frozen historical one; see tools/decision.py --baseline-kind frozen-historical), no other configuration, no
# tuning. The configuration is fixed on purpose: a generic launcher for other configurations is a separate, later piece of work.
# Submit ONLY through submit.sh (GPU type, time limit and the two baseline hashes on the command line, ledger checked first):
#   GPU_CAP_HOURS=5.4 bash $CODE/tools/origami/submit.sh $CODE/tools/origami/dev175_validation.sbatch l40s 00:28:00 BASELINE_SOURCE_SHA256=<sha256> BASELINE_DOCUMENTS_SHA256=<sha256>
# The scheduling profile is part of the frozen settings id: nothing in this script sets any engine environment variable.
#SBATCH --job-name=w0-full175
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
: "${BASELINE_SOURCE_SHA256:?BASELINE_SOURCE_SHA256 is passed by submit.sh (KEY=VALUE)}"
: "${BASELINE_DOCUMENTS_SHA256:?BASELINE_DOCUMENTS_SHA256 is passed by submit.sh (KEY=VALUE)}"
export CUDA_HOME=${EBROOTCUDA:?CUDA module not loaded}
use_vllm_env
require_clean_commit
cd "$CODE"
OUT=$BASE/full175-$GPU_TAG-$SLURM_JOB_ID
mkdir -p "$OUT"
record_device "$OUT/device.txt"
cat "$OUT/device.txt"
SETTINGS_ID=vllm-rp105-sched0-v1
CONFIG=$CODE/configs/runtime/fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0.json
JOBS175=$BASE/jobs/jobs-dev175.json
R1="vllm-$GPU_TAG-full175-r1"
R2="vllm-$GPU_TAG-full175-r2"
BASELINE_ID=replay-175
test -f "$JOBS175" && test -f "$CONFIG"
for inv in "$R1" "$R2"; do
  if [ -e "$BASE/runs/$inv/wall.json" ]; then echo "run $inv already finished: refusing to reuse its id"; exit 3; fi
done
# the config must name the fixed settings id, and the jobs file must hold the 175 recordings in their manifest order; the declared experiment is written BEFORE any run
"$VENV_VLLM/bin/python" - "$OUT/full175-manifest.json" "$CONFIG" "$SETTINGS_ID" "$JOBS175" "$SLURM_JOB_ID" "$MERALION_WORKER_GIT_COMMIT" "$R1" "$R2" "$BASELINE_ID" "$BASELINE_SOURCE_SHA256" "$BASELINE_DOCUMENTS_SHA256" <<'PYEOF'
import hashlib, json, re, subprocess, sys
from meralion_worker import config as cfg
from meralion_worker.decoders import frozen
out, config, wanted, jobs, job, commit, r1, r2, baseline_id, source_sha, documents_sha = sys.argv[1:12]
loaded = cfg.load_runtime_config(config, env={})
found = loaded.data["decoder"]["settings_id"]
if found != wanted:
    raise SystemExit(f"the config names settings {found!r}, not the fixed {wanted!r}")
names = [j["job_name"] for j in json.load(open(jobs))["jobs"]]
if len(names) != 175 or len(set(names)) != 175:
    raise SystemExit(f"the jobs file must hold exactly 175 distinct recordings, found {len(names)}")
smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
uuid = re.search(r"UUID: ((?:GPU|MIG)-[0-9a-f-]+)", smi)
manifest = {"schema": "full175-manifest-v1", "slurm_job_id": job, "worker_commit": commit, "gpu_uuid": uuid.group(1) if uuid else None, "promoted_settings_id": wanted,
            "promoted_config": config.rsplit("/", 1)[-1], "config_fingerprint": loaded.fingerprint, "settings_digest": frozen.frozen_digest(wanted), "jobs_sha256": hashlib.sha256(open(jobs, "rb").read()).hexdigest(),
            "recordings": len(names), "candidate_invocation_ids": [r1, r2],
            "baseline": {"kind": "frozen-historical", "source_sha256": source_sha, "documents_sha256": documents_sha, "invocation_ids": [baseline_id]}}
open(out, "w").write(json.dumps(manifest, indent=1) + "\n")
print(json.dumps(manifest))
PYEOF
echo "== declared: $OUT/full175-manifest.json"
# a required step: its exit code decides, and nothing after a failed required step runs (a failed first repetition is never repeated here: the evidence is kept and reported)
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
need 1_full175_r1 run_worker "$VENV_VLLM/bin/python" vllm "$R1" "$JOBS175" --config "$CONFIG"
need 2_full175_r2 run_worker "$VENV_VLLM/bin/python" vllm "$R2" "$JOBS175" --config "$CONFIG"
echo "full-175 results in $OUT and $BASE/runs/$R1 and $BASE/runs/$R2"
