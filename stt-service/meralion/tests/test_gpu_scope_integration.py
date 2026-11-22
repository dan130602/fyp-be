"""REV 4.1 A3, the parts around gpu_scope.py: the sampler (bash, fake nvidia-smi), the smoke's cross-check (fake torch) and every caller of the measurement (they receive an OBJECT, not a number)."""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
ORIGAMI = TOOLS / "origami"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(HERE))

import acceptance  # noqa: E402
import compare_backends  # noqa: E402
import full_report  # noqa: E402
import gpu_support  # noqa: E402
import synthetic  # noqa: E402
from harness import run  # noqa: E402
from test_gpu_scope import FakeNvml, HEADER, PARENT, SLICE, row  # noqa: E402

gs = gpu_support.gpu_scope


def working_bash():
    bash = shutil.which("bash")
    try:
        if bash and subprocess.run([bash, "-c", "true"], capture_output=True, timeout=20).returncode == 0:
            return bash
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


BASH = working_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no working bash")


def unix(path):
    return Path(path).as_posix()


# ---- the sampler -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def common_text():
    return (ORIGAMI / "common.sh").read_text(encoding="utf-8")


def sampler_functions():
    body = common_text()
    return body[body.index("allocated_device()"):body.index("allocated_identity()")] + body[body.index("# start sampling the ALLOCATED device"):body.index("# vLLM jobs")]


def fake_smi(tmp_path, *, fail=False):
    """A fake nvidia-smi: logs every call, prints rows to stdout and a warning to stderr (or fails with an error text)."""
    log = tmp_path / "smi.log"
    script = tmp_path / "bin" / "nvidia-smi"
    script.parent.mkdir()
    rows_file = tmp_path / "rows.txt"
    rows_file.write_text(HEADER + chr(10) + "2026/11/17 10:00:00.0, GPU, MIG-aaa, 254 MiB, 24000 MiB, 0 %" + chr(10), encoding="utf-8")
    lines = ["#!/bin/sh", f'echo "$*" >> "{unix(log)}"']
    if fail:
        lines += ['echo "Unable to determine the device handle for MIG-aaa: Not Found" >&2', "exit 6"]
    else:
        lines += ['case "$1" in', '  -L) echo "GPU 0: Fake (UUID: GPU-parent)";;', f'  *) echo "warning: scoped query on a MIG device" >&2; cat "{unix(rows_file)}";;', "esac", "exit 0"]
    script.write_text(chr(10).join(lines) + chr(10), encoding="utf-8", newline="")
    script.chmod(0o755)
    return script.parent, log


@needs_bash
@pytest.mark.parametrize("fail", [False, True])
def test_the_sampler_sends_errors_to_a_separate_file_and_never_into_the_rows(tmp_path, fail):
    bin_dir, log = fake_smi(tmp_path, fail=fail)
    csv_path = tmp_path / "gpu.csv"
    script = f'export CUDA_VISIBLE_DEVICES=MIG-aaa\n{sampler_functions()}\nstart_gpu_sampler "{unix(csv_path)}"\nwait "$SAMPLER_PID" || true\n'
    done = subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=60, env={**os.environ, "PATH": f"{unix(bin_dir)}:{os.environ['PATH']}"})
    assert done.returncode == 0, done.stderr
    rows, errors = csv_path.read_text(encoding="utf-8"), Path(str(csv_path) + ".err").read_text(encoding="utf-8")
    assert "warning" not in rows and "Unable to determine" not in rows                          # an error text can never become a row
    assert ("Unable to determine" in errors) if fail else ("warning: scoped query" in errors)
    assert ("254 MiB" in rows) is (not fail)
    calls = [line for line in log.read_text(encoding="utf-8").splitlines() if "--query-gpu" in line]
    assert calls and all("-i MIG-aaa" in line for line in calls)                                # scoped to the allocated device: no unscoped fallback
    parsed = gs.parse_sampler(rows)
    assert (len(parsed["rows"]) == 0) if fail else (len(parsed["rows"]) == 1 and parsed["malformed"] == 0)


def test_the_device_variable_is_read_never_assigned_and_the_scope_record_follows_the_sampler_stop():
    body = common_text()
    for line in body.splitlines():
        assert not re.match(r"(export\s+|unset\s+)?CUDA_VISIBLE_DEVICES(=|\s*$)", line.strip()), line
    worker = body[body.index("run_worker() {"):]
    assert worker.index("stop_gpu_sampler") < worker.index("record_gpu_scope") < worker.index("wall.json")
    assert '2>> "$1.err"' in body and "--query-gpu=timestamp,name,uuid,memory.used,memory.total,utilization.gpu" in body
    assert "gpu_scope.py" in body and "|| echo" in body                                          # a failing scope tool never fails the run
    assert "tools/gpu_scope.py" in (ORIGAMI / "sync_code.sh").read_text(encoding="utf-8")


# ---- the smoke's measurement, with a fake torch and a fake nvidia-smi -------------------------------------------------------------------------------------------------------------------------

def load_smoke():
    spec = importlib.util.spec_from_file_location("device_smoke_under_gpu_test", ORIGAMI / "device_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTorch:
    class cuda:
        @staticmethod
        def mem_get_info():
            return (16100 * 2 ** 20, 24000 * 2 ** 20)

        @staticmethod
        def memory_allocated():
            return 7000 * 2 ** 20

        @staticmethod
        def max_memory_allocated():
            return 7500 * 2 ** 20

        @staticmethod
        def memory_reserved():
            return 7800 * 2 ** 20


def fake_smi_run(sample_uuid=SLICE, sample_total=24000, sample_used=7900, fail_sample=False):
    def run_command(command, **kwargs):
        joined = " ".join(command)
        if command[:2] == ["nvidia-smi", "-L"]:
            out = f"GPU 0: NVIDIA RTX PRO 6000 (UUID: {PARENT}){chr(10)}  MIG 1g.24gb Device 0: (UUID: {SLICE}){chr(10)}"
        elif "memory.total" in joined and "noheader" in joined:
            out = "97887" + chr(10)
        elif "--query-gpu" in joined:
            if fail_sample:
                return types.SimpleNamespace(returncode=6, stdout="", stderr="not found")
            out = HEADER + chr(10) + row(uuid=sample_uuid, used=sample_used, total=sample_total) + chr(10)
        else:
            out = ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    return run_command


def test_the_smoke_reports_a_verified_slice_with_the_vocabulary_and_never_gates():
    facts = load_smoke().memory_scope_facts(run=fake_smi_run(), env={"CUDA_VISIBLE_DEVICES": SLICE}, torch_module=FakeTorch, nvml=FakeNvml())
    assert facts["gpu_memory_scope"] == "slice_verified" and facts["allocated_uuid"] == SLICE and facts["parent_uuid"] == PARENT and facts["parent_total_mib"] == 97887
    assert facts["torch_device_counters_mib"] == {"total": 24000, "used": 7900} and facts["allocator_this_process_mib"] == {"memory_allocated": 7000, "max_memory_allocated": 7500, "memory_reserved": 7800}
    vocabulary = " ".join(facts["vocabulary"].values())
    assert facts["nvml_by_uuid"]["verdict"]["scope"] == "slice_verified" and "RESERVATION" in vocabulary and "PyTorch allocator" in vocabulary


@pytest.mark.parametrize("kwargs, scope, reason", [
    (dict(sample_uuid=PARENT, sample_total=97887), "scope_unverified", "parent_identity"),
    (dict(sample_total=97887), "scope_unverified", "capacity_parent_sized"),
    (dict(sample_uuid=""), "scope_unverified", "identity_missing"),
    (dict(sample_used=20000), "scope_unverified", "numbers_disagree"),
    (dict(fail_sample=True), "unavailable", "no_valid_rows"),
])
def test_the_smokes_scope_is_unverified_or_unavailable_in_each_failure_case_and_the_numbers_are_still_reported(kwargs, scope, reason):
    facts = load_smoke().memory_scope_facts(run=fake_smi_run(**kwargs), env={"CUDA_VISIBLE_DEVICES": SLICE}, torch_module=FakeTorch, nvml=FakeNvml())
    assert facts["gpu_memory_scope"] == scope and facts["reason"] == reason and facts["torch_device_counters_mib"]["total"] == 24000


def test_the_smoke_never_changes_the_device_variable_and_does_not_import_torch_at_module_level():
    import ast
    source = (ORIGAMI / "device_smoke.py").read_text(encoding="utf-8")
    top_level = {alias.name.split(".")[0] for node in ast.parse(source).body if isinstance(node, ast.Import) for alias in node.names}
    assert "torch" not in top_level and "vllm" not in top_level and "environ[" not in source and "putenv" not in source
    env = {"CUDA_VISIBLE_DEVICES": SLICE}
    load_smoke().memory_scope_facts(run=fake_smi_run(), env=env, torch_module=FakeTorch, nvml=FakeNvml())
    assert env == {"CUDA_VISIBLE_DEVICES": SLICE}
    assert "gpu_memory" in source and '"checks": checks' in source and "memory" not in source[source.index("def plan_checks"):source.index("def expected_platform_class_from_pre")]      # reported, never one of the named checks


# ---- every caller handles the object ---------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_run_reports_carry_the_object_and_only_a_verified_scope_gives_a_number(tmp_path):
    run(synthetic.World({"alpha": dict(break_at=2)}), tmp_path, invocation_id="r", base="r")
    folder = tmp_path / "r"
    docs, _ = acceptance.scan_documents(folder / "out" / "transcripts")
    cases = (("a verified whole GPU", dict(used=40000), "device_verified", 40000),
             ("the parent's rows under a MIG allocation", dict(used=42000, allocated="MIG-aaaa-1111", parent="GPU-bbbb-2222", sampler_uuid="GPU-bbbb-2222", total=97887), "scope_unverified", None),
             ("a sampler file without a scope record", dict(used=43000, record=False), "unavailable", None))
    for label, arguments, scope, number in cases:
        (folder / gs.SCOPE_FILE).unlink(missing_ok=True)
        gpu_support.write_scope(folder, **arguments)
        for facts in (compare_backends.run_facts(docs, folder, folder / "gpu.csv")["gpu_memory"], full_report.run_facts(docs, folder)["gpu_memory"]):
            assert facts["scope"] == scope and facts["peak_mib"] == number, label
            assert facts["device_reported_peak_mib"] == arguments["used"]
            assert (number is not None) or "NOT verified" in facts["label"]                       # a number-less report says why, in words


def test_e1_selection_keeps_the_object_and_a_numeric_peak_only_for_a_verified_scope(tmp_path):
    import test_e1_selection as e1
    world = e1.E1World(tmp_path)
    ids = [run_id for run_ids in world.run_ids.values() for run_id in run_ids]
    first, other = ids[0], ids[1]
    gpu_support.write_scope(world.folder(first), used=21000)
    result = world.decide()
    assert result["runs"][first]["gpu_memory"]["scope"] == "device_verified" and result["runs"][first]["gpu_peak_mib"] == 21000
    (world.folder(first) / gs.SCOPE_FILE).unlink()
    gpu_support.write_scope(world.folder(first), used=21000, sampler_uuid="GPU-9")                      # the sampler's rows name another device than the one run-meta says was allocated
    result = world.decide()
    assert result["runs"][first]["gpu_peak_mib"] is None and result["runs"][first]["gpu_memory"]["reason"] == "identity_mismatch" and result["runs"][first]["gpu_memory"]["device_reported_peak_mib"] == 21000
    assert result["runs"][other]["gpu_memory"]["scope"] == "unavailable" and result["runs"][other]["gpu_peak_mib"] is None
    assert result["promoted"] is not None                                                          # unavailable or unverified telemetry never changes eligibility


# ---- REV 4.2 Fix 2: the sampler is stopped and reaped BEFORE its file is read or hashed -------------------------------------------------------------------------------------------------------------

FINAL_ROW = "2026/11/17 10:00:09.0, GPU, MIG-aaa, 999 MiB, 24000 MiB, 0 %"


def slow_stop_smi(tmp_path, *, ignore_term=False):
    """A fake nvidia-smi that keeps sampling; on SIGTERM it needs a moment and then writes a FINAL row (like a real sampler flushing at exit); or it ignores SIGTERM altogether."""
    script = tmp_path / "bin" / "nvidia-smi"
    script.parent.mkdir()
    final = "" if ignore_term else f"sleep 1; echo '{FINAL_ROW}'; exit 0"
    trap = "trap '' TERM" if ignore_term else f'trap "{final}" TERM'
    lines = ["#!/bin/sh", 'case "$1" in -L) echo "GPU 0: Fake (UUID: GPU-parent)"; exit 0;; esac', trap, f"echo '{HEADER}'", "echo '2026/11/17 10:00:00.0, GPU, MIG-aaa, 254 MiB, 24000 MiB, 0 %'",
             "while true; do sleep 0.2; done"]
    script.write_text(chr(10).join(lines) + chr(10), encoding="utf-8", newline="")
    script.chmod(0o755)
    return script.parent


def run_sampler_script(tmp_path, bin_dir, body, extra_env=None, timeout=60):
    """Runs `body` in bash with the sampler functions, the fake nvidia-smi first on PATH (added INSIDE bash with cygpath, never by joining a Windows PATH with a colon)."""
    nl = chr(10)
    script = 'PATH="$(cygpath -u "$STUB_BIN" 2>/dev/null || printf %s "$STUB_BIN"):$PATH"; export PATH' + nl + "export CUDA_VISIBLE_DEVICES=MIG-aaa" + nl + sampler_functions() + nl + body + nl
    return subprocess.run([BASH, "-c", script], capture_output=True, text=True, timeout=timeout, env={**os.environ, "STUB_BIN": unix(bin_dir), **(extra_env or {})})


@needs_bash
def test_stopping_the_sampler_waits_for_its_final_write_before_anything_reads_the_file(tmp_path):
    bin_dir = slow_stop_smi(tmp_path)
    csv_path = tmp_path / "gpu.csv"
    done = run_sampler_script(tmp_path, bin_dir, f'start_gpu_sampler "{unix(csv_path)}"\nsleep 1\nstop_gpu_sampler\ncat "{unix(csv_path)}"\necho "pid=[$SAMPLER_PID]"')
    assert done.returncode == 0, done.stderr
    assert FINAL_ROW in done.stdout and "pid=[]" in done.stdout                                  # the row written while the sampler was stopping is in the file when the stop returns


@needs_bash
def test_a_sampler_that_ignores_sigterm_is_killed_within_the_bound_and_a_second_stop_is_harmless(tmp_path):
    bin_dir = slow_stop_smi(tmp_path, ignore_term=True)
    csv_path = tmp_path / "gpu.csv"
    done = run_sampler_script(tmp_path, bin_dir, f'start_gpu_sampler "{unix(csv_path)}"\nsleep 1\nP=$SAMPLER_PID\nstop_gpu_sampler\nkill -0 "$P" 2>/dev/null && echo STILL_ALIVE || echo GONE\nstop_gpu_sampler\necho second=$?',
                              extra_env={"SAMPLER_STOP_STEPS": "4"}, timeout=90)
    assert done.returncode == 0, done.stderr
    assert "GONE" in done.stdout and "STILL_ALIVE" not in done.stdout and "second=0" in done.stdout


@needs_bash
def test_the_scope_record_is_made_from_the_final_sampler_file_so_the_hash_matches_the_bytes_on_disk(tmp_path):
    bin_dir = slow_stop_smi(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run-meta.json").write_text('{"gpu_uuid": "MIG-aaa", "parent_gpu_uuid": "GPU-parent"}', encoding="utf-8")
    body = f'CODE="{unix(TOOLS.parent)}"\nstart_gpu_sampler "{unix(run_dir / "gpu.csv")}"\nsleep 1\nstop_gpu_sampler\nrecord_gpu_scope "{unix(run_dir)}" "{unix(sys.executable)}"'
    done = run_sampler_script(tmp_path, bin_dir, body)
    assert done.returncode == 0, done.stderr
    record = gs.json.loads((run_dir / "gpu-scope.json").read_text(encoding="utf-8"))
    csv_bytes = (run_dir / "gpu.csv").read_bytes()
    assert b"999 MiB" in csv_bytes and record["sampler_csv_sha256"] == gs.hashlib.sha256(csv_bytes).hexdigest()
    assert record["sampler_format"]["valid_rows"] == 2                                           # the first row and the one written while stopping


def test_the_stop_function_waits_and_reaps_and_is_called_before_the_scope_record():
    body = common_text()
    stop = body[body.index("stop_gpu_sampler() {"):body.index("# record_gpu_scope <run dir>")]
    assert "kill -KILL" in stop and 'wait "$pid"' in stop and "kill -0" in stop and 'SAMPLER_PID=""' in stop and "SAMPLER_STOP_STEPS" in stop
