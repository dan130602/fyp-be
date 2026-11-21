#!/bin/bash -l
# The matched Blackwell comparison (REV 2.9 section 2 as amended by REV 3.0 A, 3.1, 3.2; "Option A"): ONE allocation on ONE MIG slice holds BOTH backends on the same ordered 175-recording development set:
# the BF16 profile vllm-rp105-sched0-v1 twice (fresh engines), its standalone CALL_064, then a FRESH Transformers reference (standalone CALL_064, then the full set). The expensive Transformers step starts only
# after vLLM has proved itself in this allocation (steps fail-stop, and a cheap structural gate on the first vLLM repetition sits before it). Every step is its own process; ids are refused if used.
# One uninterrupted attempt (--no-requeue + require_first_attempt): a preemption cancels the job, the evidence stays as it is and a fresh job is a NEW request. Fixed to this experiment: not a generic launcher.
# The controlled logits test is NOT here (login node, REV 3.2). The scheduling variable appears on ONE command line only: the stock plugin probe, which must inspect the engine in process.
#   1 preflight + declared manifest   2 apply_patches --check   3 stock and candidate plugin probes   4 live capture   5 vLLM full-175 r1   6 structural gate on r1   7 vLLM full-175 r2
#   8 vLLM CALL_064   9 Transformers CALL_064   10 Transformers full-175
# Submit ONLY through submit.sh (limit decided from the smoke's measured speed and approved first; the planned limit is 05:20:00):
#   GPU_CAP_HOURS=<approved cap> bash $CODE/tools/origami/submit.sh $CODE/tools/origami/bw_matched.sbatch blackwell 05:20:00
#SBATCH --job-name=w0-bw-matched
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
OUT=$BASE/bw-matched-$GPU_TAG-$SLURM_JOB_ID
require_first_attempt "$OUT"
mkdir -p "$OUT"
record_device "$OUT/device.txt"
cat "$OUT/device.txt"
SETTINGS_ID=vllm-rp105-sched0-v1
CANDIDATE_CONFIG=$CODE/configs/runtime/fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0.json
TOOLS=$CODE/tools/origami
JOBS175=$BASE/jobs/jobs-dev175.json
JOBS_C4=$BASE/jobs/jobs-c4.json
WINDOWS=$TOOLS/problem_windows.json
VR1="vllm-$GPU_TAG-bw175-r1"
VR2="vllm-$GPU_TAG-bw175-r2"
VC4="vllm-$GPU_TAG-bw175-c4"
TF175="tf-$GPU_TAG-bw175-1"
TFC4="tf-$GPU_TAG-bw175-c4"
test -f "$JOBS175" && test -f "$JOBS_C4" && test -f "$CANDIDATE_CONFIG" && test -f "$WINDOWS"
for inv in "$VR1" "$VR2" "$VC4" "$TF175" "$TFC4"; do
  if [ -e "$BASE/runs/$inv/wall.json" ] || [ -e "$BASE/runs/$inv/run-meta.json" ]; then echo "run $inv already started: refusing to reuse its id"; exit 3; fi
done
ALLOCATED=$(allocated_identity)
test -n "$ALLOCATED"
# the declared experiment, written BEFORE any run (the GPU identity is the ALLOCATED device: on a MIG node the slice, not the shared parent card)
"$VENV_VLLM/bin/python" - "$OUT/e2-manifest.json" "$CANDIDATE_CONFIG" "$SETTINGS_ID" "$JOBS175" "$SLURM_JOB_ID" "$MERALION_WORKER_GIT_COMMIT" "$ALLOCATED" "$VR1" "$VR2" "$VC4" "$TF175" "$TFC4" <<'PYEOF'
import hashlib, json, sys
from meralion_worker import config as cfg
from meralion_worker.decoders import frozen
out, config, wanted, jobs, job, commit, device, r1, r2, c4, tf175, tfc4 = sys.argv[1:13]
loaded = cfg.load_runtime_config(config, env={})
found = loaded.data["decoder"]["settings_id"]
if found != wanted:
    raise SystemExit(f"the config names settings {found!r}, not the fixed {wanted!r}")
names = [j["job_name"] for j in json.load(open(jobs))["jobs"]]
if len(names) != 175 or len(set(names)) != 175:
    raise SystemExit(f"the jobs file must hold exactly 175 distinct recordings, found {len(names)}")
manifest = {"schema": "e2-manifest-v1", "experiment": "bw-matched-full175", "slurm_job_id": job, "worker_commit": commit, "gpu_uuid": device, "promoted_settings_id": wanted,
            "promoted_config": config.rsplit("/", 1)[-1], "config_fingerprint": loaded.fingerprint, "settings_digest": frozen.frozen_digest(wanted),
            "jobs_sha256": hashlib.sha256(open(jobs, "rb").read()).hexdigest(), "recordings": len(names),
            "baseline_invocation_ids": {"pilot": tf175, "c4": tfc4}, "candidate_invocation_ids": [r1, r2], "candidate_c4_invocation_id": c4, "control_invocation_id": None}
open(out, "w").write(json.dumps(manifest, indent=1) + "\n")
print(json.dumps(manifest))
PYEOF
echo "== declared: $OUT/e2-manifest.json (device $ALLOCATED)"
need() {
  local name=$1
  shift
  echo "=== step $name"
  set +e
  "$@" 2>&1 | tee "$OUT/$name.log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "=== step $name exit $rc"
  echo "$name $rc" >> "$OUT/steps.txt"
  if [ "$rc" != "0" ]; then echo "required step $name failed (exit $rc): nothing after it runs"; exit "$rc"; fi
}
need 2_apply_patches_check "$VENV_VLLM/bin/python" "$CODE/tools/apply_patches.py" --check --expect patched
need 3a_probe_stock_diagnostic env VLLM_ENABLE_V1_MULTIPROCESSING=0 "$VENV_VLLM/bin/python" "$TOOLS/vllm_plugin_probe.py" --settings-id stock-vllm-package-v1 --jobs "$JOBS175" --call CALL_134 --start 24960000 --end 25440000 --out "$OUT/probe-stock.json"
need 3b_probe_candidate_profile "$VENV_VLLM/bin/python" "$TOOLS/vllm_plugin_probe.py" --settings-id "$SETTINGS_ID" --jobs "$JOBS175" --call CALL_134 --start 24960000 --end 25440000 --out "$OUT/probe-candidate.json"
need 4_capture "$VENV_VLLM/bin/python" -u "$TOOLS/vllm_windows.py" --settings-id "$SETTINGS_ID" --jobs "$JOBS175" --windows "$WINDOWS" --passes batch --capture "$OUT/capture.json" --out "$OUT/windows-capture.json"
need 5_vllm_full175_r1 run_worker "$VENV_VLLM/bin/python" vllm "$VR1" "$JOBS175" --config "$CANDIDATE_CONFIG"
# a cheap structural gate on the FIRST repetition, before the long Transformers step: every recording present, the invocation finished with exit 0, every decode ended with an end token.
# (the full structural verification is done locally by tools/decision.py; this only avoids spending the Transformers hours on a run that cannot be accepted)
need 6_structural_gate_r1 "$VENV_VLLM/bin/python" - "$BASE/runs/$VR1" <<'PYEOF'
import glob, json, os, sys
run = sys.argv[1]
summary = json.load(open(os.path.join(run, "summary.json")))
documents = sorted(glob.glob(os.path.join(run, "out", "transcripts", "*.json")))
problems = []
if summary.get("exit_code") != 0 or summary.get("failed") != 0 or summary.get("done") != summary.get("members"):
    problems.append(f"summary: {summary.get('exit_code')=} {summary.get('failed')=} {summary.get('done')=} {summary.get('members')=}")
if len(documents) != 175:
    problems.append(f"{len(documents)} documents, expected 175")
unusual = 0
for path in documents:
    block = json.load(open(path, encoding="utf-8"))["meralion"]
    rows = [w.get("stop_reason") for w in block["trace"]["windows"]] + [d["result"].get("stop_reason") for d in block["retries"]["decodes"]]
    unusual += sum(1 for reason in rows if reason != "end_token")
if unusual:
    problems.append(f"{unusual} decodes did not end with an end token")
if problems:
    sys.exit("structural gate on the first vLLM repetition FAILED: " + "; ".join(problems))
print("structural gate on the first vLLM repetition passed: 175 documents, all decodes ended with an end token")
PYEOF
need 7_vllm_full175_r2 run_worker "$VENV_VLLM/bin/python" vllm "$VR2" "$JOBS175" --config "$CANDIDATE_CONFIG"
need 8_vllm_call064 run_worker "$VENV_VLLM/bin/python" vllm "$VC4" "$JOBS_C4" --config "$CANDIDATE_CONFIG"
need 9_tf_call064 run_worker "$VENV_TF/bin/python" transformers "$TFC4" "$JOBS_C4"
need 10_tf_full175 run_worker "$VENV_TF/bin/python" transformers "$TF175" "$JOBS175"
echo "matched results in $OUT and $BASE/runs/{$VR1,$VR2,$VC4,$TFC4,$TF175}"
