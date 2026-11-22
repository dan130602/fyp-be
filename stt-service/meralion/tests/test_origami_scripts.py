"""The Stage 3 cluster scripts: syntax and the rules they must keep (no cluster is contacted)."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ORIGAMI = Path(__file__).resolve().parent.parent / "tools" / "origami"
SBATCH = sorted(ORIGAMI.glob("*.sbatch"))


def _working_bash():
    """A bash that really runs (on some Windows machines `bash` is a WSL stub without a distribution)."""
    try:
        return shutil.which("bash") is not None and subprocess.run(["bash", "-c", "true"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def text(name):
    return (ORIGAMI / name).read_text(encoding="utf-8")


def test_every_script_exists():
    assert {p.name for p in ORIGAMI.iterdir()} >= {"common.sh", "sync_code.sh", "setup_vllm_venv.sh", "probe.sbatch", "worker_tf.sbatch", "vllm_smoke.sbatch", "worker_vllm.sbatch", "vllm_smoke.py", "compare_gpu.sbatch", "submit.sh", "ledger.py"}


@pytest.mark.parametrize("path", sorted(ORIGAMI.glob("*.sh")) + SBATCH, ids=lambda p: p.name)
def test_scripts_are_valid_bash_with_lf_endings(path):
    assert b"\r" not in path.read_bytes()
    if _working_bash():
        assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


@pytest.mark.parametrize("path", SBATCH, ids=lambda p: p.name)
def test_every_sbatch_uses_the_student_qos_one_gpu_four_cpus_and_32g_within_the_qos_cap(path):
    body = path.read_text(encoding="utf-8")
    constraint = "--constraint=2080" if path.name == "fp16_smoke.sbatch" else "--constraint=rtx_pro_6000_blackwell_max"          # T0 runs on a Turing board; submit.sh passes the real type on the command line
    for directive in ("--partition=student", "--account=student", "--qos=studentqos", "--gres=gpu:1", constraint, "--cpus-per-task=4", "--mem=32G", "--output=%x-%j.log"):
        assert f"#SBATCH {directive}" in body, directive
    assert re.search(r"#SBATCH --time=\d\d:\d\d:\d\d", body)
    assert body.startswith("#!/bin/bash -l")


def test_the_total_requested_gpu_time_is_about_four_hours():
    hours = []
    for name in ("probe.sbatch", "worker_tf.sbatch", "vllm_smoke.sbatch", "worker_vllm.sbatch"):
        h, m, _ = map(int, re.search(r"--time=(\d+):(\d+):(\d+)", text(name)).groups())
        hours.append(h + m / 60)
    assert sum(hours) <= 4.0 + 0.01, hours


def test_caches_and_the_vllm_environment_live_in_scratch_and_the_offline_flags_are_set():
    common = text("common.sh")
    for name in ("PIP_CACHE_DIR", "VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "TORCHINDUCTOR_CACHE_DIR", "XDG_CACHE_HOME"):
        assert re.search(rf"export {name}=\$SCR/cache/", common), name
    assert "VENV_VLLM=$SCR/.venv-vllm" in common and "/common/scratch/" in common
    assert "HF_HUB_OFFLINE=1" in common and "TRANSFORMERS_OFFLINE=1" in common and "HF_HUB_DISABLE_TELEMETRY=1" in common and "PYTHONDONTWRITEBYTECODE=1" in common


def test_the_vllm_jobs_load_cuda_and_never_pick_another_attention_backend():
    for name in ("vllm_smoke.sbatch", "worker_vllm.sbatch"):
        body = text(name)
        assert "module load CUDA/12.8.0" in body and "CUDA_HOME" in body
        assert "VLLM_ATTENTION_BACKEND" not in body and "attention" not in body.lower().replace("attention backend", "").replace("other attention backend is tried", "")
    assert "VLLM_ATTENTION_BACKEND" not in text("common.sh") and "VLLM_ATTENTION_BACKEND" not in text("setup_vllm_venv.sh")


def test_run_ids_are_fixed_and_never_reused():
    assert "tf-pilot22-1" in text("worker_tf.sbatch") and "tf-c4-1" in text("worker_tf.sbatch") and "vllm-smoke-$GPU_TAG-c4-1" in text("vllm_smoke.sbatch") and "vllm-pilot22-1" in text("worker_vllm.sbatch")
    for name in ("worker_tf.sbatch", "vllm_smoke.sbatch", "worker_vllm.sbatch"):
        assert "refusing to reuse its id" in text(name)


def test_the_sync_script_excludes_bytecode_and_nothing_secret_is_in_any_script():
    sync = text("sync_code.sh")
    assert "--exclude='__pycache__'" in sync and "--exclude='*.pyc'" in sync and "bytecode files on the remote side" in sync
    for path in list(ORIGAMI.iterdir()):
        if path.is_file():
            body = path.read_text(encoding="utf-8").lower()
            assert "password" not in body and "sshpass" not in body and "aws_secret" not in body


def test_the_worker_run_helper_keeps_the_exit_code_and_writes_a_wall_time_file():
    common = text("common.sh")
    assert "PIPESTATUS[0]" in common and "wall.json" in common and "--summary-output" in common and "--output-dir" in common and "--state-dir" in common


# ---- Stage 3 follow-up: a failed prerequisite stops what depends on it (Codex finding 3) ---------------------------------------------------------------------------------

def test_a_failing_pip_check_is_not_suppressed():
    body = text("setup_vllm_venv.sh")
    line = [l for l in body.splitlines() if "pip check" in l and not l.lstrip().startswith("#")][0]
    assert "|| true" not in line and "exit 4" in line and body.index("pip check") < body.index("verify-environment")


def test_the_c4_reference_runs_only_after_a_clean_pilot():
    body = text("worker_tf.sbatch")
    pilot, c4 = body.index("tf-pilot22-1 \"$JOBS"), body.index("tf-c4-1 \"$JOBS")
    assert pilot < c4 and "exit $rc" in body[pilot:c4] and "rc=0" not in body and "|| rc=$?" not in body


def test_the_smoke_job_stops_before_the_worker_run_when_the_smoke_fails():
    body = text("vllm_smoke.sbatch")
    assert "set -eo pipefail" in body and body.index("vllm_smoke.py") < body.index("run_worker ") and "vllm_smoke.py\" --model-dir" in body
    smoke_line = [l for l in body.splitlines() if "vllm_smoke.py" in l and "python" in l][0]
    assert "|| true" not in smoke_line and "||" not in smoke_line
