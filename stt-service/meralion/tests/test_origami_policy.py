"""REV 3.0 A2 / REV 3.1 Amendment 2: the single-attempt policy and the ledger accounting of cancelled submissions. submit.sh runs here against FAKE sbatch / scontrol / scancel / squeue; common.sh functions run
in bash with a fake nvidia-smi. Nothing contacts a cluster.
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ORIGAMI = Path(__file__).resolve().parent.parent / "tools" / "origami"


def text(name):
    return (ORIGAMI / name).read_text(encoding="utf-8")


def load(name):
    spec = importlib.util.spec_from_file_location(name, ORIGAMI / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ledger = load("ledger")


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
    """Git Bash on Windows needs /c/... (a drive-letter colon would split PATH)."""
    posix = Path(path).as_posix()
    return f"/{posix[0].lower()}{posix[2:]}" if len(posix) > 2 and posix[1] == ":" else posix


# ---- the ledger: annotate never changes the allocation ---------------------------------------------------------------------------------------------------------------------

def test_annotate_adds_status_and_reason_and_never_changes_the_allocation(tmp_path):
    path = tmp_path / "ledger.json"
    ledger.record(path, 10.0, "02:00:00", "11", "smoke.sbatch", "blackwell")
    ledger.record(path, 10.0, "00:30:00", "12", "smoke.sbatch", "blackwell")
    before = ledger.load(path)
    spent, hours = ledger.spent_seconds(before), ledger.remaining_seconds(before, 10.0)
    ledger.annotate(path, "11", "cancelled", "Requeue=1 despite --no-requeue")
    after = ledger.load(path)
    entry = next(job for job in after["jobs"] if job["job"] == "11")
    assert entry["status"] == "cancelled" and entry["reason"] == "Requeue=1 despite --no-requeue" and entry["annotated"] and entry["allocated_seconds"] == 7200
    assert len(after["jobs"]) == 2 and ledger.spent_seconds(after) == spent == 9000 and ledger.remaining_seconds(after, 10.0) == hours
    assert "status" not in next(job for job in after["jobs"] if job["job"] == "12")


def test_annotate_of_an_unknown_job_is_an_error_and_writes_nothing(tmp_path):
    path = tmp_path / "ledger.json"
    ledger.record(path, 10.0, "00:30:00", "12", "smoke.sbatch", "blackwell")
    before = path.read_bytes()
    with pytest.raises(SystemExit, match="cannot annotate"):
        ledger.annotate(path, "99", "cancelled", "x")
    assert path.read_bytes() == before


def test_the_command_line_annotate_joins_the_reason_words(tmp_path):
    path = tmp_path / "ledger.json"
    ledger.record(path, 10.0, "00:30:00", "12", "smoke.sbatch", "blackwell")
    assert ledger.main(["annotate", str(path), "12", "cancelled", "requeue", "state", "unknown"]) == 0
    assert ledger.load(path)["jobs"][0]["reason"] == "requeue state unknown"


# ---- submit.sh against fake scheduler commands -------------------------------------------------------------------------------------------------------------------------------------

class Cluster:
    def __init__(self, tmp_path):
        self.base = tmp_path / "base"
        (self.base / "code" / "tools" / "origami").mkdir(parents=True)
        shutil.copy(ORIGAMI / "ledger.py", self.base / "code" / "tools" / "origami" / "ledger.py")
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.log = tmp_path / "calls.log"
        self.script = tmp_path / "job.sbatch"
        self.script.write_text("#!/bin/bash\n", encoding="utf-8")
        self.ledger = self.base / "worker-v0-gpu-ledger.json"
        self.fake("python3", f'exec "{unix(sys.executable)}" "$@"')
        self.fake("sbatch", f'echo "sbatch $*" >> "{unix(self.log)}"\n[ "${{FAKE_SBATCH:-ok}}" = fail ] && {{ echo "sbatch: error" >&2; exit 1; }}\necho 4242')
        self.fake("scancel", f'echo "scancel $*" >> "{unix(self.log)}"')
        self.fake("squeue", 'echo "${FAKE_SQUEUE:-}"')
        self.fake("scontrol", 'case "${FAKE_SCONTROL:-ok}" in\n'
                              '  ok) echo "JobId=4242 JobName=x Priority=1 Requeue=0 Restarts=0 BatchFlag=1";;\n'
                              '  requeue1) echo "JobId=4242 JobName=x Requeue=1 Restarts=0";;\n'
                              '  nofield) echo "JobId=4242 JobName=x Restarts=0";;\n'
                              '  empty) ;;\n'
                              '  fail) echo "scontrol: error" >&2; exit 1;;\nesac')

    def fake(self, name, body):
        path = self.bin / name
        path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def submit(self, *args, env=None, cap="10.0"):
        exports = " ".join(f"{k}={v}" for k, v in (env or {}).items())
        command = f'export PATH="{unix(self.bin)}:$PATH" W0_BASE="{unix(self.base)}" GPU_CAP_HOURS={cap} {exports}; bash "{unix(ORIGAMI / "submit.sh")}" "{unix(self.script)}" {" ".join(args)}'
        return subprocess.run([BASH, "-c", command], capture_output=True, text=True, timeout=120)

    def calls(self):
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def entries(self):
        return ledger.load(self.ledger)["jobs"] if self.ledger.is_file() else []


@pytest.fixture
def cluster(tmp_path):
    return Cluster(tmp_path)


@needs_bash
def test_the_normal_path_submits_with_no_requeue_records_and_verifies_requeue_zero(cluster):
    done = cluster.submit("blackwell", "00:30:00")
    assert done.returncode == 0, done.stderr + done.stdout
    sbatch = [c for c in cluster.calls() if c.startswith("sbatch")]
    assert len(sbatch) == 1 and "--no-requeue" in sbatch[0] and "--gres=gpu:rtx_pro_6000_blackwell_max:1" in sbatch[0] and "--time=00:30:00" in sbatch[0]
    entry = cluster.entries()
    assert [(e["job"], e["allocated_seconds"], "status" in e) for e in entry] == [("4242", 1800, False)] and not [c for c in cluster.calls() if c.startswith("scancel")]


@needs_bash
@pytest.mark.parametrize("mode, fragment", [("requeue1", "Requeue=1 despite --no-requeue"), ("fail", "could not be verified"), ("nofield", "no Requeue field"), ("empty", "could not be verified")])
def test_a_job_whose_requeue_state_is_not_verified_zero_is_cancelled_annotated_and_still_counted(cluster, mode, fragment):
    done = cluster.submit("blackwell", "01:00:00", env={"FAKE_SCONTROL": mode})
    assert done.returncode == 6, done.stderr + done.stdout
    assert [c for c in cluster.calls() if c.startswith("scancel")] == ["scancel 4242"]
    entries = cluster.entries()
    assert len(entries) == 1 and entries[0]["job"] == "4242" and entries[0]["allocated_seconds"] == 3600                    # recorded BEFORE the inspection, counted in full after the cancellation
    assert entries[0]["status"] == "cancelled" and fragment in entries[0]["reason"] and "stays counted" in done.stdout


@needs_bash
def test_the_cancelled_state_is_checked_and_printed(cluster):
    done = cluster.submit("blackwell", "00:30:00", env={"FAKE_SCONTROL": "requeue1", "FAKE_SQUEUE": "COMPLETING"})
    assert done.returncode == 6 and "queue state now: COMPLETING" in done.stdout


@needs_bash
def test_a_ledger_write_failure_cancels_the_job_and_says_it_must_be_added_by_hand(cluster):
    cluster.ledger.mkdir()                                                                                    # a directory where the ledger file should be: the write fails
    done = cluster.submit("blackwell", "00:30:00")
    assert done.returncode == 5 and "LEDGER WRITE FAILED" in done.stdout and "4242" in done.stdout and "by hand" in done.stdout
    assert [c for c in cluster.calls() if c.startswith("scancel")] == ["scancel 4242"]


@needs_bash
def test_a_failing_sbatch_submits_and_records_nothing(cluster):
    done = cluster.submit("blackwell", "00:30:00", env={"FAKE_SBATCH": "fail"})
    assert done.returncode == 4 and "nothing was submitted and nothing was recorded" in done.stdout
    assert cluster.entries() == [] and not [c for c in cluster.calls() if c.startswith("scancel")]


@needs_bash
def test_a_request_beyond_the_cap_is_refused_before_sbatch_is_called(cluster):
    done = cluster.submit("blackwell", "05:00:00", cap="1.0")
    assert done.returncode == 3 and cluster.calls() == [] and cluster.entries() == []


@needs_bash
def test_the_2080_type_is_accepted_and_an_unknown_type_is_refused(cluster):
    done = cluster.submit("2080", "00:30:00")
    sbatch = [c for c in cluster.calls() if c.startswith("sbatch")][0]
    assert done.returncode == 0 and "--gres=gpu:2080:1" in sbatch and "--constraint=2080" in sbatch and "GPU_TAG=2080" in sbatch
    refused = cluster.submit("4090", "00:30:00")
    assert refused.returncode == 2 and "unknown gpu type 4090" in refused.stdout and "2080" in refused.stdout


def test_the_script_order_is_check_then_sbatch_then_record_then_inspect_then_summary():
    body = text("submit.sh")
    order = [body.index(marker) for marker in ('ledger.py" check', "sbatch --parsable --no-requeue", 'ledger.py" record', "INFO=$(scontrol show job", 'ledger.py" annotate', 'ledger.py" show')]
    assert order == sorted(order) and "W0_BASE" in body and "set -euo pipefail" in body
    assert "Requeue=0" in body and "exit 5" in body and "exit 6" in body and "exit 4" in body


# ---- common.sh: first attempt only, the allocated identity, the run metadata --------------------------------------------------------------------------------------------------

def function_text(name):
    body = text("common.sh")
    start = body.index(f"{name}() ")
    end = body.index("\n}\n", start) + 3
    return body[start:end]


def fake_smi(tmp_path, output):
    folder = tmp_path / "smi"
    folder.mkdir(exist_ok=True)
    script = folder / "nvidia-smi"
    script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{output}\nEOF\n", encoding="utf-8", newline="\n")
    script.chmod(0o755)
    shim = folder / "python3"
    shim.write_text(f'#!/bin/sh\nexec "{unix(sys.executable)}" "$@"\n', encoding="utf-8", newline="\n")
    shim.chmod(0o755)
    return folder


MIG_SMI = "GPU 0: NVIDIA RTX PRO 6000 Blackwell (UUID: GPU-11111111-aaaa)\n  MIG 1g.24gb     Device  0: (UUID: MIG-22222222-bbbb)"


def run_bash(tmp_path, script, env_assignments="", smi=MIG_SMI):
    folder = fake_smi(tmp_path, smi)
    return subprocess.run([BASH, "-c", f'export PATH="{unix(folder)}:$PATH"; {env_assignments}\n{script}'], capture_output=True, text=True, timeout=120)


@needs_bash
@pytest.mark.parametrize("assignments, expected", [
    ("export CUDA_VISIBLE_DEVICES=MIG-22222222-bbbb", "MIG-22222222-bbbb"),                           # the slice, never the parent card
    ("export CUDA_VISIBLE_DEVICES=GPU-33333333-cccc,GPU-44444444", "GPU-33333333-cccc"),
    ("export CUDA_VISIBLE_DEVICES=0", "GPU-11111111-aaaa"),                                           # an index: the first UUID nvidia-smi lists
    ("unset CUDA_VISIBLE_DEVICES", "GPU-11111111-aaaa"),
])
def test_the_allocated_identity_is_the_slice_when_the_device_variable_names_one(tmp_path, assignments, expected):
    done = run_bash(tmp_path, f'{function_text("allocated_identity")}\n{assignments}\nallocated_identity')
    assert done.returncode == 0 and done.stdout.strip() == expected, done.stderr


@needs_bash
def test_a_second_attempt_is_refused_before_anything_is_written(tmp_path):
    out = tmp_path / "out"
    fn = function_text("require_first_attempt")
    fresh = run_bash(tmp_path, f'{fn}\nrequire_first_attempt "{unix(out)}"; echo rc=$?')
    assert "rc=0" in fresh.stdout and not out.exists()
    restarted = run_bash(tmp_path, f'{fn}\nSLURM_RESTART_COUNT=1 require_first_attempt "{unix(out)}"; echo rc=$?')
    assert "rc=3" in restarted.stdout and "restarted by the scheduler" in restarted.stdout and not out.exists()
    out.mkdir()
    existing = run_bash(tmp_path, f'{fn}\nrequire_first_attempt "{unix(out)}"; echo rc=$?')
    assert "rc=3" in existing.stdout and "already exists" in existing.stdout
    assert text("common.sh").count("require_first_attempt()") == 1


@needs_bash
def test_run_worker_refuses_to_overwrite_a_first_attempts_metadata(tmp_path):
    base = tmp_path / "base"
    (base / "runs" / "inv1").mkdir(parents=True)
    meta = base / "runs" / "inv1" / "run-meta.json"
    meta.write_text('{"first": "attempt"}', encoding="utf-8")
    script = f'BASE="{unix(base)}"\nrequire_clean_commit() {{ return 0; }}\nwrite_run_meta() {{ echo "WRITTEN"; }}\n{function_text("run_worker")}\nrun_worker py backend inv1 jobs.json; echo rc=$?'
    done = run_bash(tmp_path, script)
    assert "rc=3" in done.stdout and "run-meta.json already exists" in done.stdout and "WRITTEN" not in done.stdout and meta.read_text(encoding="utf-8") == '{"first": "attempt"}'
    assert not (base / "runs" / "inv1" / "wall.json").exists()


def run_meta_source():
    body = text("common.sh")
    start = body.index("write_run_meta() {")
    block = body[start:body.index(chr(10) + "}" + chr(10), start)]
    marker = "<<'PYEOF'" + chr(10)
    return block[block.index(marker) + len(marker):block.index(chr(10) + "PYEOF")]


@pytest.mark.parametrize("restart, cvd, identity", [("0", "MIG-22222222-bbbb", "MIG-22222222-bbbb"), ("2", "MIG-22222222-bbbb", "MIG-22222222-bbbb"), ("0", "0", "GPU-11111111-aaaa")])
def test_run_meta_records_the_allocated_slice_the_parent_card_the_restart_count_and_the_build_controls(tmp_path, monkeypatch, restart, cvd, identity):
    """The Python of write_run_meta, run directly with a fake nvidia-smi (the laptop may have a real one that a bash PATH cannot shadow for a native Python)."""
    folder = tmp_path / "run"
    folder.mkdir()
    real_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, stdout=MIG_SMI, stderr="")
        return real_run(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", fake_run)
    for key, value in dict(CUDA_VISIBLE_DEVICES=cvd, SLURM_RESTART_COUNT=restart, SLURM_JOB_ID="777", MERALION_WORKER_GIT_COMMIT="abc", MAX_JOBS="4", OMP_NUM_THREADS="4", UNRELATED="1").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "argv", ["-", str(folder), "vllm", "inv1", sys.executable, "--flag"])
    exec(compile(run_meta_source(), "write_run_meta", "exec"), {"__name__": "__main__"})
    meta = json.loads((folder / "run-meta.json").read_text(encoding="utf-8"))
    assert meta["gpu_uuid"] == identity and meta["parent_gpu_uuid"] == "GPU-11111111-aaaa" and meta["cvd_first_entry"] == cvd
    assert meta["restart_count"] == int(restart) and meta["attempt"] == 1 and any("MIG 1g.24gb" in line for line in meta["mig_profile"]) and meta["slurm_job_id"] == "777"
    assert meta["meralion_env"]["MAX_JOBS"] == "4" and meta["meralion_env"]["OMP_NUM_THREADS"] == "4" and "UNRELATED" not in meta["meralion_env"]
    assert meta["worker_arguments"] == ["--flag"] and meta["worker_git_commit"] == "abc"


# ---- sync_code.sh carries the patch files and checks them on both sides --------------------------------------------------------------------------------------------------------

def test_sync_code_lists_the_patch_files_and_compares_their_hashes_on_both_sides():
    body = text("sync_code.sh")
    tar_line = next(line for line in body.splitlines() if line.startswith("tar "))
    shipped = next(line for line in body.splitlines() if line.startswith("SHIPPED=")).split("=", 1)[1].strip('"')
    assert "$SHIPPED" in tar_line                                                              # the tar list and the byte-identity guard read ONE list
    for path in ("meralion_worker", "configs", "patches", "tools/apply_patches.py", "tools/env_identity.py", "tools/audio_fingerprint.py", "tools/make_jobs.py", "tools/origami"):
        assert f" {path} " in f" {shipped} ", path
    assert "patches/manifest.json patches/vllm-0.16.0-mig-single-device.patch tools/apply_patches.py" in body and "MISMATCH" in body and "exit 5" in body
    assert (ORIGAMI.parent.parent / "patches" / "vllm-0.16.0-mig-single-device.patch").is_file()


@needs_bash
def test_the_tar_list_of_sync_code_really_contains_the_patch_files(tmp_path):
    body = text("sync_code.sh")
    tar_line = next(line for line in body.splitlines() if line.startswith("tar "))
    names = next(line for line in body.splitlines() if line.startswith("SHIPPED=")).split("=", 1)[1].strip('"').split()
    root = ORIGAMI.parent.parent
    listing = subprocess.run([BASH, "-c", f'cd "{unix(root)}" && tar --exclude=__pycache__ --exclude="*.pyc" -czf - {" ".join(names)} | tar -tzf -'], capture_output=True, text=True, timeout=120)
    assert listing.returncode == 0, listing.stderr
    members = set(listing.stdout.split())
    assert {"patches/manifest.json", "patches/vllm-0.16.0-mig-single-device.patch", "tools/apply_patches.py", "tools/origami/device_smoke.py" if (ORIGAMI / "device_smoke.py").exists() else "tools/origami/submit.sh"} <= members


# ---- the three job scripts of REV 3.0 (text and structure; nothing runs) ------------------------------------------------------------------------------------------------------------------

NEW_JOBS = ("slice_smoke.sbatch", "fp16_smoke.sbatch", "bw_matched.sbatch")


def code_lines(name):
    return [line for line in text(name).splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_every_new_job_requires_a_clean_commit_one_attempt_and_a_declared_gpu_tag_and_refuses_used_ids():
    for name in NEW_JOBS:
        body = text(name)
        assert body.startswith("#!/bin/bash -l") and ': "${GPU_TAG:?' in body and "require_clean_commit" in body and 'require_first_attempt "$OUT"' in body
        code = chr(10).join(code_lines(name))
        assert code.index("require_clean_commit") < code.index("require_first_attempt") < code.index("mkdir -p \"$OUT\"") and "already started: refusing to reuse its id" in body
        assert "set -eo pipefail" in body and "nothing after it runs" in body and "submit.sh" in body[:2500]
        assert not [line for line in code_lines(name) if line.startswith("export CUDA_VISIBLE_DEVICES") or "unset CUDA_VISIBLE_DEVICES" in line]     # the allocated device is never rewritten


def test_the_new_jobs_use_their_fixed_profile_and_config_and_set_no_engine_variable_outside_the_one_diagnostic_probe_line():
    fixed = {"slice_smoke.sbatch": ("vllm-rp105-sched0-v1", "vllm-rp105-sched0.json"), "bw_matched.sbatch": ("vllm-rp105-sched0-v1", "vllm-rp105-sched0.json"),
             "fp16_smoke.sbatch": ("vllm-rp105-sched0-fp16-v1", "vllm-rp105-sched0-fp16.json")}
    for name, (settings_id, config) in fixed.items():
        body = text(name)
        assert f"SETTINGS_ID={settings_id}" in body and config in body
    for name in NEW_JOBS:
        lines = [line for line in code_lines(name) if "VLLM_ENABLE_V1_MULTIPROCESSING" in line]
        if name == "bw_matched.sbatch":
            assert len(lines) == 1 and "vllm_plugin_probe.py" in lines[0] and "--settings-id stock-vllm-package-v1" in lines[0] and "need 3a_probe_stock_diagnostic env " in lines[0]
        else:
            assert lines == []
        assert "VLLM_ATTENTION_BACKEND" not in text(name) and "export VLLM" not in text(name)


def test_the_logits_test_is_not_in_the_gpu_jobs_of_this_revision_but_the_older_jobs_keep_it():
    for name in NEW_JOBS:
        assert "plugin_logits_test.py" not in "\n".join(code_lines(name)), name
    for name in ("d1_diagnostics.sbatch", "e2_acceptance.sbatch"):
        assert "plugin_logits_test.py" in text(name), name


def test_the_smoke_jobs_run_their_steps_in_the_planned_order_and_classify_a_failure():
    order = {"slice_smoke.sbatch": ["2_apply_patches_check", "3_device_smoke", "4_windows_capture", "5_worker_call064", "6_worker_call103"],
             "fp16_smoke.sbatch": ["1_driver_supports_cuda_12_8", "2_device_smoke", "3_windows_capture", "4_worker_call064", "5_worker_call103"]}
    for name, steps in order.items():
        body = text(name)
        positions = [body.index(f"need {step} ") for step in steps]
        assert positions == sorted(positions) and "device_smoke.py\" classify" in body and "--settings-id \"$SETTINGS_ID\"" in body
    assert "--expect-capability 12.0 --expect-platform NonNvmlCudaPlatform" in text("slice_smoke.sbatch") and "--passes batch,batch,single" in text("slice_smoke.sbatch")
    assert "--expect-capability 7.5 --require-no-empty-text" in text("fp16_smoke.sbatch") and "apply_patches" not in text("fp16_smoke.sbatch")           # no patch on a whole GPU


def test_the_matched_job_starts_the_expensive_transformers_steps_only_after_vllm_and_its_structural_gate():
    body = text("bw_matched.sbatch")
    steps = ["2_apply_patches_check", "3a_probe_stock_diagnostic", "3b_probe_candidate_profile", "4_capture", "5_vllm_full175_r1", "6_structural_gate_r1", "7_vllm_full175_r2", "8_vllm_call064", "9_tf_call064", "10_tf_full175"]
    positions = [body.index(f"need {step} ") for step in steps]
    assert positions == sorted(positions)
    assert '"$VENV_VLLM/bin/python" vllm "$VR1"' in body and '"$VENV_TF/bin/python" transformers "$TF175"' in body and body.count("--config \"$CANDIDATE_CONFIG\"") >= 3
    assert '"e2-manifest-v1"' in body and "allocated_identity" in body and body.index("e2-manifest.json") < body.index("need 2_apply_patches_check")           # declared BEFORE any run, with the slice identity
    assert "Transformers hours" in body and "structural gate on the first vLLM repetition FAILED" in body


@needs_bash
@pytest.mark.parametrize("name", NEW_JOBS)
def test_the_new_scripts_are_valid_bash_with_lf_endings(name):
    path = ORIGAMI / name
    assert b"\r" not in path.read_bytes()
    assert subprocess.run([BASH, "-n", unix(path)], capture_output=True, text=True).returncode == 0


@needs_bash
def test_the_structural_gate_python_passes_a_complete_run_and_fails_each_defect(tmp_path):
    body = text("bw_matched.sbatch")
    marker = "<<'PYEOF'\n"
    start = body.index("need 6_structural_gate_r1")
    code = body[body.index(marker, start) + len(marker):body.index("\nPYEOF", body.index(marker, start))]

    def make(root, documents=175, summary=None, reason="end_token"):
        (root / "out" / "transcripts").mkdir(parents=True)
        (root / "summary.json").write_text(json.dumps(summary or {"exit_code": 0, "failed": 0, "done": 175, "members": 175}), encoding="utf-8")
        for number in range(documents):
            block = {"trace": {"windows": [{"stop_reason": "end_token"}]}, "retries": {"decodes": [{"result": {"stop_reason": reason if number == 0 else "end_token"}}]}}
            (root / "out" / "transcripts" / f"CALL_{number:03d}.json").write_text(json.dumps({"meralion": block}), encoding="utf-8")
        return root

    def gate(root):
        script = tmp_path / "gate.py"
        script.write_text(code, encoding="utf-8")
        return subprocess.run([sys.executable, str(script), str(root)], capture_output=True, text=True)
    assert gate(make(tmp_path / "ok")).returncode == 0
    for label, kwargs in (("missing", dict(documents=174)), ("failed", dict(summary={"exit_code": 1, "failed": 1, "done": 174, "members": 175})), ("reason", dict(reason="max_new_tokens"))):
        done = gate(make(tmp_path / label, **kwargs))
        assert done.returncode != 0 and "structural gate on the first vLLM repetition FAILED" in done.stderr, label
