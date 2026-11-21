"""The tracked vLLM compatibility patch (REV 3.0 A3): the committed patch and manifest agree, the applier is context-checked and reversible, and the patched selection logic does what the plan says.

The real original `vllm/platforms/cuda.py` lives in the cluster's .venv-vllm and is not in the repository. Tests that need it read it from MERALION_VLLM_CUDA_ORIGINAL (a local copy) and are skipped
otherwise; every other test uses a fixture rebuilt from the patch's own old side.
"""
import ast
import hashlib
import importlib.util
import json
import os
import re
import shutil
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
TOOLS = PACKAGE_ROOT / "tools"
PATCHES = PACKAGE_ROOT / "patches"


def load_tool():
    spec = importlib.util.spec_from_file_location("apply_patches", TOOLS / "apply_patches.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ap = load_tool()
MANIFEST = json.loads((PATCHES / "manifest.json").read_text(encoding="utf-8"))
ENTRY = MANIFEST["patches"][0]
PATCH_TEXT = (PATCHES / ENTRY["patch_file"]).read_text(encoding="utf-8")
ORIGINAL_COPY = os.environ.get("MERALION_VLLM_CUDA_ORIGINAL")


def old_side(patch_text):
    """The old side of every hunk, padded so the hunk's line numbers are right: a fixture the patch applies to."""
    hunk = ap.parse_hunks(patch_text)[0]
    lines = [f"# filler {n}" for n in range(1, hunk["old_start"])]
    lines += [text for kind, text in hunk["lines"] if kind in (" ", "-")]
    return "\n".join(lines) + "\n"


FIXTURE = old_side(PATCH_TEXT)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixture_env(tmp_path, *, version="0.16.0", original=FIXTURE, patched=None):
    """A fake site-packages with vllm/platforms/cuda.py = original, a dist-info of that version, and a manifest + patch describing the fixture."""
    site = tmp_path / "site"
    (site / "vllm" / "platforms").mkdir(parents=True)
    (site / "vllm" / "platforms" / "cuda.py").write_text(original, encoding="utf-8", newline="")
    (site / f"vllm-{version}.dist-info").mkdir()
    (site / f"vllm-{version}.dist-info" / "METADATA").write_text(f"Metadata-Version: 2.4\nName: vllm\nVersion: {version}\n", encoding="utf-8")
    patches = tmp_path / "patches"
    patches.mkdir()
    patch_bytes = PATCH_TEXT.encode("utf-8")
    (patches / "p.patch").write_bytes(patch_bytes)
    new = ap.apply_unified(FIXTURE, PATCH_TEXT) if patched is None else patched
    entry = {"id": "t", "target": "vllm/platforms/cuda.py", "target_version": "0.16.0", "patch_file": "p.patch", "original_sha256": sha(FIXTURE), "patched_sha256": sha(new),
             "patch_sha256": hashlib.sha256(patch_bytes).hexdigest()}
    manifest = patches / "manifest.json"
    manifest.write_text(json.dumps({"patches": [entry]}), encoding="utf-8")
    return site, manifest, entry, patches


def idle():
    return ""


# ---- the committed patch and manifest ---------------------------------------------------------------------------------------------------------------------------------------------

def test_the_committed_patch_file_matches_the_manifest_hash():
    assert hashlib.sha256((PATCHES / ENTRY["patch_file"]).read_bytes()).hexdigest() == ENTRY["patch_sha256"]
    assert ENTRY["target"] == "vllm/platforms/cuda.py" and ENTRY["target_version"] == "0.16.0" and b"\r" not in (PATCHES / ENTRY["patch_file"]).read_bytes()


def test_the_patch_is_kept_byte_exact_by_git_attributes():
    attributes = (PACKAGE_ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "patches/*.patch -text" in attributes


def test_the_patch_applies_to_its_own_old_side_and_the_result_defines_the_helper_and_the_new_selection():
    new = ap.apply_unified(FIXTURE, PATCH_TEXT)
    assert "def _single_mig_device_in_process() -> bool:" in new and "if nvml_available and not _single_mig_device_in_process()" in new
    assert "CudaPlatform = NvmlCudaPlatform if nvml_available else NonNvmlCudaPlatform" in FIXTURE and "CudaPlatform = NvmlCudaPlatform if nvml_available else NonNvmlCudaPlatform" not in new


@pytest.mark.skipif(not (ORIGINAL_COPY and Path(ORIGINAL_COPY).is_file()), reason="a local copy of the original cuda.py is not provided (MERALION_VLLM_CUDA_ORIGINAL)")
def test_against_the_real_original_the_hashes_and_the_patched_result_match_the_manifest():
    original = Path(ORIGINAL_COPY).read_bytes()
    assert hashlib.sha256(original).hexdigest() == ENTRY["original_sha256"]
    assert hashlib.sha256(ap.apply_unified(original.decode("utf-8"), PATCH_TEXT).encode("utf-8")).hexdigest() == ENTRY["patched_sha256"]


def selection(patched_text, *, nvml_available, visible, launcher):
    """Run the patched helper and the final selection expression with faked names: which platform class would be chosen."""
    start = patched_text.index("def _single_mig_device_in_process")
    end = patched_text.index("CudaPlatform.log_warnings()")
    environment = {k: v for k, v in (("CUDA_VISIBLE_DEVICES", visible), ("VLLM_ENABLE_V1_MULTIPROCESSING", launcher)) if v is not None}
    namespace = {"os": type("OS", (), {"environ": environment}), "NvmlCudaPlatform": "Nvml", "NonNvmlCudaPlatform": "NonNvml", "nvml_available": nvml_available}
    exec(patched_text[start:end], namespace)
    return namespace["CudaPlatform"]


@pytest.mark.parametrize("nvml, visible, launcher, expected", [
    (True, "MIG-abc", "0", "NonNvml"),             # the patched case: one MIG device, in-process engine
    (True, "MIG-abc", None, "Nvml"),               # a MIG device but the default multiprocess engine: unchanged behaviour
    (True, "MIG-abc", "1", "Nvml"),
    (True, "GPU-abc", "0", "Nvml"),                # a GPU UUID is NOT handled (a separate decision)
    (True, "0", "0", "Nvml"),                      # a numeric index keeps NVML
    (True, "", "0", "Nvml"),
    (True, None, "0", "Nvml"),                     # unset
    (True, "MIG-a,MIG-b", "0", "Nvml"),            # more than one device: not handled
    (True, "MIG-a,", "0", "Nvml"),                 # not exactly one entry
    (False, "0", None, "NonNvml"),                 # no NVML at all keeps the unpatched fallback
    (False, "MIG-abc", "0", "NonNvml"),
])
def test_the_patched_selection_for_every_environment(nvml, visible, launcher, expected):
    patched = ap.apply_unified(FIXTURE, PATCH_TEXT)
    assert selection(patched, nvml_available=nvml, visible=visible, launcher=launcher) == expected


def test_the_unpatched_selection_ignores_the_environment():
    unpatched = FIXTURE + "\nCudaPlatform.log_warnings()\n"
    start = unpatched.index("CudaPlatform = ")
    for nvml, expected in ((True, "Nvml"), (False, "NonNvml")):
        namespace = {"NvmlCudaPlatform": "Nvml", "NonNvmlCudaPlatform": "NonNvml", "nvml_available": nvml}
        exec(unpatched[start:unpatched.index("CudaPlatform.log_warnings()")], namespace)
        assert namespace["CudaPlatform"] == expected


# ---- the applier ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_a_hunk_that_does_not_match_the_file_is_refused_and_a_wrong_count_is_refused():
    with pytest.raises(ap.PatchError, match="does not match"):
        ap.apply_unified(FIXTURE.replace("nvml_available", "nvml_availableX"), PATCH_TEXT)
    broken = re.sub(r"^@@ -(\d+),(\d+)", lambda m: f"@@ -{m.group(1)},{int(m.group(2)) + 1}", PATCH_TEXT, count=1, flags=re.MULTILINE)
    with pytest.raises(ap.PatchError, match="old lines"):
        ap.apply_unified(FIXTURE, broken)
    with pytest.raises(ap.PatchError, match="no hunks"):
        ap.apply_unified(FIXTURE, "--- a\n+++ b\n")


def test_check_reports_original_patched_and_unknown_states(tmp_path):
    site, manifest, entry, patches = fixture_env(tmp_path)
    assert ap.check_entry(site, entry, patches, "original")["ok"] is True
    assert ap.check_entry(site, entry, patches, "patched")["ok"] is False
    (site / "vllm" / "platforms" / "cuda.py").write_text("something else", encoding="utf-8")
    result = ap.check_entry(site, entry, patches, "patched")
    assert result["state"] == "unknown" and result["ok"] is False and any("unknown" in p for p in result["problems"])


def test_a_different_installed_vllm_version_fails_the_check(tmp_path):
    site, manifest, entry, patches = fixture_env(tmp_path, version="0.17.0")
    result = ap.check_entry(site, entry, patches, "original")
    assert result["ok"] is False and any("0.17.0" in p for p in result["problems"])


STATES = ("RUNNING", "PENDING", "COMPLETING", "SUSPENDED", "CONFIGURING", "REQUEUED", "STOPPED", "RESIZING", "SIGNALING")


def snapshot(site):
    """Every file under the fake site-packages with its bytes: a refusal must leave all of it untouched (target, .orig and temporary files)."""
    return {str(path.relative_to(site)): path.read_bytes() for path in sorted(Path(site).rglob("*")) if path.is_file()}


def patched_env(tmp_path):
    """An environment in the PATCHED state with its backup, as it is after a successful --apply."""
    site, manifest, entry, patches = fixture_env(tmp_path)
    ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)
    return site, manifest, entry, patches


def test_apply_needs_the_idle_confirmation_and_an_idle_queue_and_writes_nothing_otherwise(tmp_path):
    site, manifest, entry, patches = fixture_env(tmp_path)
    before = snapshot(site)
    with pytest.raises(ap.PatchError, match="confirm-env-idle"):
        ap.apply_entry(site, entry, patches, confirm_env_idle=False, squeue=idle)
    for line in ("1|w0-e2|RUNNING", "2|w0-smoke|PENDING"):
        with pytest.raises(ap.PatchError, match="w0- jobs"):
            ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=lambda line=line: line)
    assert snapshot(site) == before and not (site / "vllm" / "platforms" / "cuda.py.orig").exists()


def test_a_failing_squeue_means_the_environment_is_not_shown_to_be_idle(tmp_path):
    site, manifest, entry, patches = fixture_env(tmp_path)

    def broken():
        raise ap.PatchError("squeue failed")
    with pytest.raises(ap.PatchError, match="squeue failed"):
        ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=broken)
    assert (site / "vllm" / "platforms" / "cuda.py").read_text(encoding="utf-8") == FIXTURE
    site2, manifest2, entry2, patches2 = patched_env(tmp_path / "b")
    before = snapshot(site2)
    with pytest.raises(ap.PatchError, match="squeue failed"):
        ap.revert_entry(site2, entry2, patches2, confirm_env_idle=True, squeue=broken)
    assert snapshot(site2) == before


# ---- REV 4.1 A2: ONE idle policy for --apply and --revert ----------------------------------------------------------------------------------------------------------------------------

def test_revert_without_the_confirmation_refuses_and_writes_nothing(tmp_path):
    """Codex finding 2, reproduced: revert_entry used to write immediately, with no flag and no queue check."""
    site, manifest, entry, patches = patched_env(tmp_path)
    before = snapshot(site)
    with pytest.raises(ap.PatchError, match="confirm-env-idle"):
        ap.revert_entry(site, entry, patches, confirm_env_idle=False, squeue=idle)
    assert snapshot(site) == before
    assert ap.inspect_entry(site, entry, patches)["state"] == "patched"


@pytest.mark.parametrize("operation", ["apply", "revert"])
@pytest.mark.parametrize("state", STATES)
def test_a_w0_job_in_any_state_blocks_apply_and_revert_and_leaves_every_file_untouched(tmp_path, operation, state):
    site, manifest, entry, patches = patched_env(tmp_path) if operation == "revert" else fixture_env(tmp_path)
    before = snapshot(site)
    queue = f"77|w0-bw-matched|{state}"
    call = ap.revert_entry if operation == "revert" else ap.apply_entry
    with pytest.raises(ap.PatchError, match="w0- jobs"):
        call(site, entry, patches, confirm_env_idle=True, squeue=lambda: queue)
    assert snapshot(site) == before


@pytest.mark.parametrize("operation", ["apply", "revert"])
def test_a_reviewed_job_does_not_unblock_a_w0_job(tmp_path, operation):
    site, manifest, entry, patches = patched_env(tmp_path) if operation == "revert" else fixture_env(tmp_path)
    call = ap.revert_entry if operation == "revert" else ap.apply_entry
    with pytest.raises(ap.PatchError, match="w0- jobs"):
        call(site, entry, patches, confirm_env_idle=True, reviewed_jobs=["77"], squeue=lambda: "77|w0-x|RUNNING")


@pytest.mark.parametrize("operation", ["apply", "revert"])
def test_another_listed_job_blocks_until_it_is_named_and_a_stale_acknowledgement_is_refused(tmp_path, operation):
    site, manifest, entry, patches = patched_env(tmp_path) if operation == "revert" else fixture_env(tmp_path)
    call = ap.revert_entry if operation == "revert" else ap.apply_entry
    before = snapshot(site)
    queue = "367308|jupyter-notebook-8999|RUNNING"
    with pytest.raises(ap.PatchError, match="not been reviewed"):
        call(site, entry, patches, confirm_env_idle=True, squeue=lambda: queue)
    with pytest.raises(ap.PatchError, match="not in the queue"):
        call(site, entry, patches, confirm_env_idle=True, reviewed_jobs=["367308", "999"], squeue=lambda: queue)         # a stale id in addition
    with pytest.raises(ap.PatchError, match="not been reviewed"):
        call(site, entry, patches, confirm_env_idle=True, reviewed_jobs=["367308"], squeue=lambda: queue + "\n5|manual-job|PENDING")      # one reviewed, one not
    assert snapshot(site) == before
    result = call(site, entry, patches, confirm_env_idle=True, reviewed_jobs=["367308"], squeue=lambda: queue)
    assert result["action"] in ("applied", "reverted")
    assert result["idle_check"]["reviewed_jobs"] == ["367308"] and "human acknowledgement" in result["idle_check"]["note"]
    assert [job["id"] for job in result["idle_check"]["queue"]] == ["367308"]


def test_the_queue_parser_keeps_names_with_spaces_and_bars_and_refuses_a_malformed_line():
    jobs = ap.parse_queue("12|my job with spaces|RUNNING\n13|odd|name|PENDING\n\n14_3|array|SUSPENDED\n")
    assert [(j["id"], j["name"], j["state"]) for j in jobs] == [("12", "my job with spaces", "RUNNING"), ("13", "odd|name", "PENDING"), ("14_3", "array", "SUSPENDED")]
    for bad in ("12 w0-x RUNNING", "12|w0-x", "|w0-x|RUNNING", "12|w0-x|"):
        with pytest.raises(ap.PatchError, match="cannot parse"):
            ap.parse_queue(bad)


def test_a_job_name_with_a_space_that_starts_with_w0_still_blocks(tmp_path):
    site, manifest, entry, patches = fixture_env(tmp_path)
    with pytest.raises(ap.PatchError, match="w0- jobs"):
        ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=lambda: "9|w0-smoke test|RUNNING")


def test_an_unparseable_queue_refuses_apply_and_revert(tmp_path):
    site, manifest, entry, patches = fixture_env(tmp_path)
    with pytest.raises(ap.PatchError, match="cannot parse"):
        ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=lambda: "garbage")
    site2, manifest2, entry2, patches2 = patched_env(tmp_path / "b")
    with pytest.raises(ap.PatchError, match="cannot parse"):
        ap.revert_entry(site2, entry2, patches2, confirm_env_idle=True, squeue=lambda: "garbage")


def test_a_revert_of_a_file_that_is_already_original_needs_no_guard(tmp_path):
    site, manifest, entry, patches = fixture_env(tmp_path)
    called = []
    result = ap.revert_entry(site, entry, patches, confirm_env_idle=False, squeue=lambda: called.append(1) or "")
    assert result["action"] == "already original" and not called


def test_an_apply_of_a_file_that_is_already_patched_needs_no_guard(tmp_path):
    site, manifest, entry, patches = patched_env(tmp_path)
    called = []
    result = ap.apply_entry(site, entry, patches, confirm_env_idle=False, squeue=lambda: called.append(1) or "")
    assert result["action"] == "already patched" and not called


def test_the_squeue_command_uses_the_account_user_not_the_environment_variable(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command

        class Result:
            returncode, stdout, stderr = 0, "", ""
        return Result()
    monkeypatch.delenv("USER", raising=False)                                  # an unset $USER used to give `squeue -u ""`
    monkeypatch.setattr(ap.getpass, "getuser", lambda: "someone")
    monkeypatch.setattr(ap.subprocess, "run", fake_run)
    assert ap.run_squeue() == ""
    assert seen["command"] == ["squeue", "-h", "-u", "someone", "-o", "%i|%j|%T"]


def test_the_command_line_revert_needs_the_confirmation_and_accepts_reviewed_jobs(tmp_path, monkeypatch, capsys):
    site, manifest, entry, patches = patched_env(tmp_path)
    assert ap.main(["--revert", "--env", str(site), "--manifest", str(manifest)]) == 3                          # no --confirm-env-idle
    assert ap.inspect_entry(site, entry, patches)["state"] == "patched"
    monkeypatch.setattr(ap, "run_squeue", lambda: "367308|jupyter-notebook-8999|RUNNING")
    assert ap.main(["--revert", "--confirm-env-idle", "--env", str(site), "--manifest", str(manifest)]) == 3     # an unreviewed job
    assert ap.inspect_entry(site, entry, patches)["state"] == "patched"
    assert ap.main(["--revert", "--confirm-env-idle", "--reviewed-job", "367308", "--env", str(site), "--manifest", str(manifest)]) == 0
    assert ap.inspect_entry(site, entry, patches)["state"] == "original"
    assert '"reviewed_jobs"' in capsys.readouterr().out


def test_apply_refuses_an_unknown_version_an_unknown_original_and_a_wrong_patch_file(tmp_path):
    site, manifest, entry, patches = fixture_env(tmp_path, version="0.17.0")
    with pytest.raises(ap.PatchError, match="installed is 0.17.0"):
        ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)
    site2, manifest2, entry2, patches2 = fixture_env(tmp_path / "b", original=FIXTURE + "# changed\n")
    with pytest.raises(ap.PatchError, match="state 'unknown'"):
        ap.apply_entry(site2, entry2, patches2, confirm_env_idle=True, squeue=idle)
    site3, manifest3, entry3, patches3 = fixture_env(tmp_path / "c")
    (patches3 / "p.patch").write_text(PATCH_TEXT + "# tampered\n", encoding="utf-8")
    with pytest.raises(ap.PatchError, match="patch file"):
        ap.apply_entry(site3, entry3, patches3, confirm_env_idle=True, squeue=idle)


def test_apply_writes_the_backup_once_replaces_atomically_verifies_the_hash_and_is_idempotent(tmp_path):
    site, manifest, entry, patches = fixture_env(tmp_path)
    target = site / "vllm" / "platforms" / "cuda.py"
    result = ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)
    assert result["action"] == "applied" and result["state"] == "patched" and sha(target.read_text(encoding="utf-8")) == entry["patched_sha256"]
    backup = Path(str(target) + ".orig")
    assert backup.read_text(encoding="utf-8") == FIXTURE and not list(target.parent.glob("*.tmp-apply"))
    again = ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)
    assert again["action"] == "already patched" and backup.read_text(encoding="utf-8") == FIXTURE


def test_apply_refuses_when_the_patched_result_would_not_match_the_manifest(tmp_path):
    site, manifest, entry, patches = fixture_env(tmp_path, patched="not what the patch produces\n")
    with pytest.raises(ap.PatchError, match="patched sha256"):
        ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)
    assert (site / "vllm" / "platforms" / "cuda.py").read_text(encoding="utf-8") == FIXTURE


def test_the_replacement_goes_through_a_temporary_file_in_the_same_directory(tmp_path, monkeypatch):
    site, manifest, entry, patches = fixture_env(tmp_path)
    seen = []
    real_replace = os.replace

    def spy(source, destination):
        seen.append((Path(source).parent, Path(destination).parent, Path(source).name))
        real_replace(source, destination)
    monkeypatch.setattr(ap.os, "replace", spy)
    ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)
    assert seen and all(src_dir == dst_dir for src_dir, dst_dir, _ in seen) and all(name.endswith(".tmp-apply") for _, _, name in seen)


def test_revert_restores_the_original_bytes_and_refuses_without_the_recorded_backup(tmp_path):
    site, manifest, entry, patches = fixture_env(tmp_path)
    target = site / "vllm" / "platforms" / "cuda.py"
    ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)
    assert ap.revert_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)["action"] == "reverted" and target.read_text(encoding="utf-8") == FIXTURE
    assert ap.revert_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)["action"] == "already original"
    ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)
    Path(str(target) + ".orig").write_text("not the original", encoding="utf-8")
    with pytest.raises(ap.PatchError, match="not the recorded original"):
        ap.revert_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)
    Path(str(target) + ".orig").unlink()
    with pytest.raises(ap.PatchError, match="backup"):
        ap.revert_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)


def test_the_command_line_check_and_exit_codes(tmp_path, capsys):
    site, manifest, entry, patches = fixture_env(tmp_path)
    assert ap.main(["--check", "--expect", "original", "--env", str(site), "--manifest", str(manifest)]) == 0
    assert ap.main(["--check", "--env", str(site), "--manifest", str(manifest)]) == 1                      # default expectation is patched
    assert ap.main(["--check", "--env", str(site), "--manifest", str(tmp_path / "missing.json")]) == 3
    assert ap.main(["--apply", "--env", str(site), "--manifest", str(manifest)]) == 3                      # no --confirm-env-idle
    out = capsys.readouterr().out
    assert '"state": "original"' in out


def test_the_runtime_identity_reads_the_committed_manifest_and_classifies_the_installed_file(tmp_path, monkeypatch):
    from meralion_worker import runtime_identity as ri
    site, manifest, entry, patches = fixture_env(tmp_path)
    monkeypatch.syspath_prepend(str(site))
    (site / "vllm" / "__init__.py").write_text("", encoding="utf-8")
    (site / "vllm" / "platforms" / "__init__.py").write_text("", encoding="utf-8")
    found = ri.patches_identity(manifest)
    assert found[0]["state"] == "original" and found[0]["patch_sha256"] == entry["patch_sha256"]
    ap.apply_entry(site, entry, patches, confirm_env_idle=True, squeue=idle)
    assert ri.patches_identity(manifest)[0]["state"] == "patched"
    committed = ri.patches_identity(PATCHES / "manifest.json")
    assert committed and committed[0]["target"] == "vllm/platforms/cuda.py"


def test_the_tool_uses_the_standard_library_only():
    tree = ast.parse((TOOLS / "apply_patches.py").read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= {"argparse", "getpass", "hashlib", "importlib", "json", "os", "re", "subprocess", "sys", "pathlib"}
