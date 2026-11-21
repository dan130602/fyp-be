#!/bin/bash
# Shared settings for the Stage 3 Origami scripts (sourced, never run). Everything large lives in SCRATCH; home keeps only the existing model and audio.
SCR=/common/scratch/users/e/YOUR_USERNAME/meralion
BASE=$SCR/runs/worker-v0
CODE=$BASE/code
PILOT=$HOME/meralion-pilot
MODEL=$PILOT/model
VENV_TF=$PILOT/.venv-cu128
VENV_VLLM=$SCR/.venv-vllm
export MERALION_MODEL_DIR=$MODEL
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$CODE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

# pip, compilation and inference caches go to scratch (home has a quota)
export PIP_CACHE_DIR=$SCR/cache/pip
export XDG_CACHE_HOME=$SCR/cache/xdg
export VLLM_CACHE_ROOT=$SCR/cache/vllm
export TRITON_CACHE_DIR=$SCR/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$SCR/cache/inductor
export FLASHINFER_WORKSPACE_BASE=$SCR/cache/flashinfer
export CUDA_CACHE_PATH=$SCR/cache/cuda
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$CUDA_CACHE_PATH"

# the device Slurm allocated: the first entry of CUDA_VISIBLE_DEVICES (a GPU- or MIG- UUID, or an index); empty when the variable is unset
allocated_device() { local first=${CUDA_VISIBLE_DEVICES%%,*}; echo "$first"; }

# The identity of the device Slurm ALLOCATED: the first entry of CUDA_VISIBLE_DEVICES when it is a MIG- or GPU- UUID (on a MIG node `nvidia-smi -L` lists the shared PARENT card first, so the first UUID there
# would name the card, not the slice); otherwise (an index, or unset) the first UUID of `nvidia-smi -L`. Prints nothing when none can be found.
allocated_identity() {
  local first=${CUDA_VISIBLE_DEVICES%%,*}
  case "$first" in
    MIG-*|GPU-*) echo "$first"; return 0 ;;
  esac
  nvidia-smi -L 2>/dev/null | grep -o 'UUID: \(GPU\|MIG\)-[0-9a-f-]*' | head -1 | sed 's/^UUID: //' || true
}

# An acceptance or diagnostic run is ONE uninterrupted attempt (REV 3.0 A2): refuse, before anything is created or written, when Slurm restarted this job or the output folder already exists.
require_first_attempt() {
  if [ "${SLURM_RESTART_COUNT:-0}" != "0" ]; then echo "refusing to run: this job was restarted by the scheduler (SLURM_RESTART_COUNT=$SLURM_RESTART_COUNT); a restarted run is not acceptance evidence"; return 3; fi
  if [ -e "${1:?output folder}" ]; then echo "refusing to run: $1 already exists (a second attempt never reuses or overwrites a first attempt's files)"; return 3; fi
}

# write which physical device this job got (host, CUDA_VISIBLE_DEVICES, nvidia-smi -L) into $1
record_device() {
  { echo "host: $(hostname)"; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"; nvidia-smi -L; } > "$1" 2>&1
}

# start sampling the ALLOCATED device only (memory, utilisation) every 5 s into $1, with a header naming the device; the sampler's pid is left in SAMPLER_PID.
# REV 4.1 A3: stdout (the rows) goes to the CSV, stderr to $1.err, so an error text can never become a row; the query stays scoped with -i (CUDA_VISIBLE_DEVICES is only READ, never rewritten) and there is
# NO unscoped fallback: an unscoped query on a MIG node reports the PARENT card's memory.total. Whether the rows describe the allocated device is decided afterwards by tools/gpu_scope.py (record_gpu_scope).
start_gpu_sampler() {
  local dev
  dev=$(allocated_device)
  { echo "# allocated device: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; nvidia-smi -L; } > "$1" 2> "$1.err"
  nvidia-smi ${dev:+-i "$dev"} --query-gpu=timestamp,name,uuid,memory.used,memory.total,utilization.gpu --format=csv -l 5 >> "$1" 2>> "$1.err" &
  SAMPLER_PID=$!
}
# REV 4.2: `kill` is asynchronous, so the sampler is STOPPED and REAPED before anything reads or hashes its file: SIGTERM, a bounded wait (SAMPLER_STOP_STEPS x 0.5 s, default 20), SIGKILL if it is still alive,
# then `wait` (which reaps it, so the file is closed and complete). Idempotent (the sbatch scripts also call it from an EXIT trap) and never fails the caller.
stop_gpu_sampler() {
  local pid=${SAMPLER_PID:-} steps=0
  [ -n "$pid" ] || return 0
  kill "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null && [ "$steps" -lt "${SAMPLER_STOP_STEPS:-20}" ]; do sleep 0.5; steps=$((steps + 1)); done
  if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi
  wait "$pid" 2>/dev/null || true
  SAMPLER_PID=""
}

# record_gpu_scope <run dir> <python>: write <run dir>/gpu-scope.json (REV 4.1 A3): did the sampler's rows describe the device Slurm ALLOCATED to this run (identity first, then capacity)? A measurement note: it never
# fails the run, and an unavailable or unverified scope only means the memory figure is reported as device-reported and not as a slice measurement.
record_gpu_scope() {
  "$2" "$CODE/tools/gpu_scope.py" --run-meta "$1/run-meta.json" --csv "$1/gpu.csv" --out "$1/gpu-scope.json" >> "$1/gpu-scope.log" 2>&1 || echo "gpu_scope could not run for $1 (see gpu-scope.log): the memory scope stays unavailable"
}

# vLLM jobs: FlashInfer builds its kernels at run time and starts `ninja` by NAME, so the vLLM environment's bin directory must be on PATH (the venv is not activated in a batch job). This is a
# PATH fix only: no vLLM setting, sampling parameter or constructor default changes. The first vLLM trial on a whole L40S failed with FileNotFoundError: 'ninja' before this.
use_vllm_env() {
  export PATH="$VENV_VLLM/bin:$PATH"
  command -v ninja >/dev/null || { echo "ninja is not on PATH after use_vllm_env"; return 1; }
  command -v nvcc >/dev/null || { echo "nvcc is not on PATH (load the CUDA module first)"; return 1; }
}

# The worker commit of the code that was synced (sync_code.sh writes CODE_COMMIT): exported so every document records it. Acceptance runs refuse "unknown" and "-dirty".
if [ -f "$CODE/CODE_COMMIT" ]; then MERALION_WORKER_GIT_COMMIT=$(tr -d '\n' < "$CODE/CODE_COMMIT"); else MERALION_WORKER_GIT_COMMIT=unknown; fi
export MERALION_WORKER_GIT_COMMIT
require_clean_commit() {
  case "$MERALION_WORKER_GIT_COMMIT" in
    ""|unknown|*-dirty) echo "refusing to run: the synced code has no clean recorded commit (found '$MERALION_WORKER_GIT_COMMIT'); sync a clean committed tree with sync_code.sh"; return 1 ;;
  esac
}

# write_run_meta <run dir> <backend> <invocation id> <python> [worker arguments...]: what this run actually was (commit, allocation, physical GPU, environment, effective arguments)
write_run_meta() {
  python3 - "$@" <<'PYEOF'
import hashlib, json, os, platform, re, subprocess, sys
run_dir, backend, invocation, python = sys.argv[1:5]
smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
uuid = re.search(r"UUID: ((?:GPU|MIG)-[0-9a-f-]+)", smi)                                  # the FIRST UUID: on a MIG node this is the shared parent card
first_visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
allocated = first_visible if re.fullmatch(r"(?:GPU|MIG)-[0-9a-f-]+", first_visible) else (uuid.group(1) if uuid else None)       # the device Slurm allocated (the slice on a MIG node)
mig_lines = [line.strip() for line in smi.splitlines() if "MIG" in line]
freeze = subprocess.run([python, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
meta = {"worker_git_commit": os.environ.get("MERALION_WORKER_GIT_COMMIT"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "host": platform.node(), "gpu_uuid": allocated,
        "parent_gpu_uuid": uuid.group(1) if uuid else None, "cvd_first_entry": first_visible or None, "mig_profile": mig_lines, "restart_count": int(os.environ.get("SLURM_RESTART_COUNT") or 0), "attempt": 1,
        "nvidia_smi_L": smi.strip().splitlines(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "backend": backend, "invocation_id": invocation, "python": python,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(), "worker_arguments": sys.argv[5:],
        "vllm_enable_v1_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "unset"), "meralion_env": {k: v for k, v in os.environ.items() if k.startswith(("MERALION_", "HF_HUB", "TRANSFORMERS_OFFLINE", "FLASHINFER", "VLLM_", "MAX_JOBS", "OMP_", "MKL_", "OPENBLAS_", "TORCHINDUCTOR_"))}}
open(os.path.join(run_dir, "run-meta.json"), "w").write(json.dumps(meta, indent=1, default=str) + "\n")
PYEOF
}

# run_worker <python> <backend> <invocation id> <jobs file> [extra worker arguments...]: one worker invocation with ITS OWN GPU sampler (one file per run, first row = idle before the model loads).
# The exit code is kept; a wall-time file and run-meta.json are written next to the outputs. An acceptance run needs a clean recorded worker commit.
run_worker() {
  local py=$1 backend=$2 inv=$3 jobs=$4
  shift 4
  require_clean_commit || return 1
  local dir=$BASE/runs/$inv
  if [ -e "$dir/run-meta.json" ]; then echo "refusing to run $inv: $dir/run-meta.json already exists (a second attempt never overwrites a first attempt's metadata)"; return 3; fi
  mkdir -p "$dir"
  write_run_meta "$dir" "$backend" "$inv" "$py" "$@"
  start_gpu_sampler "$dir/gpu.csv"
  sleep 3
  local started ended rc
  started=$(date +%s)
  set +e
  "$py" -u -m meralion_worker run --invocation-id "$inv" --backend "$backend" --jobs "$jobs" --model-dir "$MODEL" \
      --output-dir "$dir/out" --state-dir "$dir/state" --summary-output "$dir/summary.json" "$@" 2>&1 | tee "$dir/worker.log"
  rc=${PIPESTATUS[0]}
  set -e
  ended=$(date +%s)
  stop_gpu_sampler
  record_gpu_scope "$dir" "$py"
  echo "{\"wall_seconds\": $((ended - started)), \"exit_code\": $rc}" > "$dir/wall.json"
  echo "run $inv backend=$backend exit=$rc wall=$((ended - started))s"
  return $rc
}
