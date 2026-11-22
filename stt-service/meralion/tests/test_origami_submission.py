"""Submission mechanics of the Stage 3 follow-up: GPU type and time on the command line, the GPU-time ledger, the single-GPU sampler, the same-GPU comparison job."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

ORIGAMI = Path(__file__).resolve().parent.parent / "tools" / "origami"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ORIGAMI / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ledger = load("ledger")


def text(name):
    return (ORIGAMI / name).read_text(encoding="utf-8")


# ---- shell variables are never expanded inside #SBATCH lines -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(ORIGAMI.glob("*.sbatch")), ids=lambda p: p.name)
def test_no_sbatch_directive_contains_a_shell_variable(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#SBATCH"):
            assert "$" not in line and "{" not in line, line


def test_the_wrapper_passes_gpu_type_constraint_and_time_on_the_actual_command_line():
    body = text("submit.sh")
    command = [line for line in body.splitlines() if "sbatch --parsable" in line][0]
    assert '--gres="gpu:$GRES:1"' in command and '--constraint="$FEATURE"' in command and '--time="$LIMIT"' in command and '--export="$EXPORTS"' in command
    for gpu, gres in (("blackwell", "rtx_pro_6000_blackwell_max"), ("l40s", "l40s"), ("a40", "a40"), ("3090", "3090")):
        assert re.search(rf"{gpu}\)\s+GRES={gres};\s+FEATURE={gres}", body), gpu
    assert "GPU_TAG=$GPU" in body                                                             # the jobs read the type from the environment, never from a directive


def test_the_wrapper_checks_the_ledger_before_submitting_and_records_after():
    body = text("submit.sh")
    assert body.index("ledger.py\" check") < body.index("sbatch --parsable") < body.index("ledger.py\" record") and "CAP_HOURS=${GPU_CAP_HOURS:-5.0}" in body
    assert "set -euo pipefail" in body


# ---- the ledger ----------------------------------------------------------------------------------------------------------------------------------------------------------

def test_time_limits_are_parsed():
    assert ledger.seconds_of("00:10:00") == 600 and ledger.seconds_of("2:18:00") == 8280 and ledger.seconds_of("1-00:00:00") == 86400 and ledger.seconds_of("45:00") == 2700


def test_the_ledger_counts_every_allocation_in_full_and_refuses_what_exceeds_the_remainder(tmp_path):
    path = tmp_path / "ledger.json"
    for job, limit in (("1", "00:10:00"), ("2", "00:10:00"), ("3", "00:10:00")):               # three probes, whatever happened to them
        ledger.record(path, 5.0, limit, job, "probe.sbatch", "l40s")
    ledger.record(path, 5.0, "00:42:00", "4", "vllm_smoke.sbatch", "l40s")
    assert ledger.spent_seconds(ledger.load(path)) == 3 * 600 + 2520
    ok, remaining, requested = ledger.check(path, 5.0, "02:18:00")
    assert ok and remaining == 18000 - 4320 and requested == 8280
    ledger.record(path, 5.0, "02:18:00", "5", "compare_gpu.sbatch", "l40s")
    ok, remaining, _ = ledger.check(path, 5.0, "01:31:00")
    assert remaining == 18000 - 4320 - 8280 == 5400 and not ok                                 # 5460 s requested against 5400 s left: refused
    assert ledger.check(path, 5.0, "01:30:00")[0] is True
    with pytest.raises(SystemExit, match="refused"):
        ledger.record(path, 5.0, "01:31:00", "6", "x.sbatch", "l40s")
    assert len(ledger.load(path)["jobs"]) == 5                                                 # a refused job is not recorded


def test_exactly_the_cap_is_allowed_and_one_second_more_is_not(tmp_path):
    path = tmp_path / "ledger.json"
    assert ledger.check(path, 5.0, "05:00:00")[0] is True and ledger.check(path, 5.0, "05:00:01")[0] is False


def test_the_command_line_interface_returns_the_verdict(tmp_path, capsys):
    path = str(tmp_path / "ledger.json")
    assert ledger.main(["check", path, "5.0", "00:10:00"]) == 0
    assert ledger.main(["check", path, "5.0", "06:00:00"]) == 3
    assert ledger.main(["record", path, "5.0", "00:10:00", "7", "probe.sbatch", "3090"]) == 0
    assert ledger.main(["show", path, "5.0"]) == 0 and json.loads(capsys.readouterr().out.split("\n", 2)[2])["spent_hours"] == pytest.approx(0.167, abs=0.001)


# ---- the sampler samples the allocated device only ----------------------------------------------------------------------------------------------------------------------

def test_the_sampler_selects_only_the_allocated_device_and_names_it():
    common = text("common.sh")
    assert 'nvidia-smi ${dev:+-i "$dev"} --query-gpu=timestamp,name,uuid,memory.used,memory.total,utilization.gpu' in common
    assert "allocated_device() { local first=${CUDA_VISIBLE_DEVICES%%,*}; echo \"$first\"; }" in common and "# allocated device:" in common and "nvidia-smi -L" in common
    assert "record_device()" in common


def test_the_probe_requires_exactly_one_gpu_row_reports_mig_and_checks_vllms_own_device_mapping():
    body = text("probe.sbatch")
    assert 'test "$ROWS" = "1"' in body and "a MIG slice" in body and "a whole GPU (no MIG)" in body
    assert "current_platform.get_device_capability()" in body and "exit $VLLM_RC" in body


def test_the_peak_memory_parser_reads_the_new_layout_and_ignores_the_device_header(tmp_path):
    compare = load_tool("gpu_scope")                                                          # the raw reader moved to gpu_scope (REV 4.1 A3); its number is NOT a slice measurement by itself
    gpu = tmp_path / "gpu.csv"
    gpu.write_text("# allocated device: CUDA_VISIBLE_DEVICES=GPU-abc\nGPU 0: NVIDIA L40S (UUID: GPU-abc)\n"
                   "timestamp, name, uuid, memory.used [MiB], memory.total [MiB], utilization.gpu [%]\n"
                   "2026/11/16 10:00:00.1, NVIDIA L40S, GPU-abc, 3000 MiB, 46068 MiB, 10 %\n2026/11/16 10:00:05.1, NVIDIA L40S, GPU-abc, 21000 MiB, 46068 MiB, 99 %\n", encoding="utf-8")
    assert compare.raw_peak_mib(gpu) == 21000
    old = tmp_path / "old.csv"
    old.write_text("timestamp, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]\n2026/11/16 10:00:00.1, GPU, 3000 MiB, 24192 MiB, 10 %\n", encoding="utf-8")
    assert compare.raw_peak_mib(old) == 3000


def load_tool(name):
    import sys
    tools = ORIGAMI.parent
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(name, tools / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- the same-GPU comparison job -------------------------------------------------------------------------------------------------------------------------------------------

def test_the_comparison_job_runs_both_backends_and_both_standalone_references_in_one_allocation_in_order():
    body = text("compare_gpu.sbatch")
    order = [body.index(marker) for marker in ('transformers "tf-$GPU_TAG-pilot22-1"', 'transformers "tf-$GPU_TAG-c4-1"', 'vllm "vllm-$GPU_TAG-pilot22-1"', 'vllm "vllm-$GPU_TAG-c4-1"')]
    assert order == sorted(order) and body.count("start_gpu_sampler") == 1                  # one sampler, one allocation, one physical GPU
    assert '$VENV_TF/bin/python' in body and '$VENV_VLLM/bin/python' in body                 # two separate pinned environments
    assert "VLLM_ATTENTION_BACKEND" not in body and "refusing to reuse its id" in body and ': "${GPU_TAG:?' in body


def test_the_comparison_job_stops_after_any_failed_step_and_defaults_to_a_tiny_time_limit():
    body = text("compare_gpu.sbatch")
    assert body.count("nothing after it runs") == 3 and "set -eo pipefail" in body
    assert "#SBATCH --time=00:30:00" in body                                                   # a direct sbatch cannot spend real GPU time; submit.sh sets the real limit
    assert "submit.sh" in body.splitlines()[2] or "submit.sh" in body[:700]


# ---- the FlashInfer remedy: ninja and nvcc must be on PATH for every vLLM job ------------------------------------------------------------------------------------------------

def test_every_vllm_job_puts_the_vllm_environment_on_the_path_after_loading_cuda_and_changes_no_vllm_setting():
    common = text("common.sh")
    assert 'export PATH="$VENV_VLLM/bin:$PATH"' in common and "command -v ninja" in common and "command -v nvcc" in common
    for name in ("vllm_smoke.sbatch", "worker_vllm.sbatch", "compare_gpu.sbatch"):
        body = text(name)
        assert body.index("module load CUDA/12.8.0") < body.index("use_vllm_env") < body.index("run_worker") if "run_worker" in body else True
        assert "VLLM_ATTENTION_BACKEND" not in body and "use_vllm_env" in body
    assert "gpu_memory_utilization" not in common and "max_model_len" not in common and "VLLM_ATTENTION_BACKEND" not in common


def test_the_path_helper_actually_finds_a_fake_ninja_and_nvcc(tmp_path):
    """Run the helper in bash with fake executables, so the failure mode of the first L40S trial (ninja not found) is reproduced and fixed."""
    import shutil
    import subprocess
    bash = shutil.which("bash")
    if not bash or subprocess.run([bash, "-c", "true"], capture_output=True).returncode != 0:
        pytest.skip("no working bash")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    ninja = venv_bin / "ninja"
    ninja.write_text("#!/bin/sh\necho ninja\n")
    ninja.chmod(0o755)
    cuda_bin = tmp_path / "cuda"
    cuda_bin.mkdir()
    nvcc = cuda_bin / "nvcc"
    nvcc.write_text("#!/bin/sh\necho nvcc\n")
    nvcc.chmod(0o755)
    body = text("common.sh")
    helper = body[body.index("use_vllm_env()"):body.index("# run_worker <python>")]
    def unix(path):                                      # Git Bash on Windows needs /c/... in PATH (a drive-letter colon would split it)
        posix = path.as_posix()
        return f"/{posix[0].lower()}{posix[2:]}" if len(posix) > 2 and posix[1] == ":" else posix

    base = f'VENV_VLLM="{unix(venv_bin.parent)}"; {helper}\n'
    without = subprocess.run([bash, "-c", base + f'export PATH="{unix(cuda_bin)}:/usr/bin:/bin"; command -v ninja'], capture_output=True, text=True)
    assert without.returncode != 0                                                             # before: ninja is not found by name
    fixed = subprocess.run([bash, "-c", base + f'export PATH="{unix(cuda_bin)}:/usr/bin:/bin"; use_vllm_env && command -v ninja'], capture_output=True, text=True)
    assert fixed.returncode == 0 and fixed.stdout.strip().endswith("ninja")
    missing_nvcc = subprocess.run([bash, "-c", base + 'export PATH="/usr/bin:/bin"; use_vllm_env'], capture_output=True, text=True)
    assert missing_nvcc.returncode != 0 and "nvcc" in missing_nvcc.stdout
