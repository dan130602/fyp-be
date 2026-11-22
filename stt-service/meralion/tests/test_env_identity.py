"""tools/env_identity.py (REV 3.2 Amendment 7: a portable environment comparison instead of a `pip freeze` text hash) and tools/audio_fingerprint.py (REV 3.2 Amendment 4/7)."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ei = load("env_identity")
af = load("audio_fingerprint")
WHEEL = "939911f3d5856a307adf723d042eac21842afc40a008e94e5f74245af5455b9a"
MANIFEST = {"artifact": {"sha256": WHEEL}}


def reference(**over):
    data = {"schema": "environment-reference-v1", "pins": {"vllm": "0.16.0", "torch": "2.9.1+cu128", "numpy": "2.2.6", "soundfile": "0.14.0", "meralion-3-asr": "0.1.0", "pip": "24.0"},
            "meralion_artifact_sha256": WHEEL, "torch": {"version": "2.9.1+cu128", "cuda": "12.8"},
            "audio": {"soundfile": "0.14.0", "librosa": "1.0.1", "soxr": "1.1.0", "numpy": "2.2.6", "scipy": "1.18.1", "libsndfile": "1.2.2"},
            "patches": [{"target": "vllm/platforms/cuda.py", "state": "patched", "installed_sha256": "p" * 64}],
            "informational": {"python": "3.12.8", "platform": "Linux-cluster", "machine": "x86_64", "system": "Linux", "freeze_text_sha256": "a" * 64, "never_compared": True}, "ignored_packages": ["pip", "setuptools", "wheel"]}
    data.update(over)
    return data


def current(**over):
    here = json.loads(json.dumps(reference()))
    here["informational"] = {"python": "3.12.3", "platform": "Linux-ec2-ubuntu24", "machine": "x86_64", "system": "Linux", "freeze_text_sha256": "b" * 64}      # a different machine, path, Python patch version and freeze text
    here["pins"]["pip"] = "25.1"                                                                                                        # installer tooling is ignored
    here["patches"] = [{"target": "vllm/platforms/cuda.py", "state": "original", "installed_sha256": "o" * 64}]
    here.update(over)
    return here


def check(here=None, ref=None, manifest=MANIFEST, expect="original", verify=0):
    return ei.compare(ref or reference(), here or current(), manifest, expect, verify)


def test_pep_503_normalisation():
    assert ei.normalise("Meralion_3.ASR") == "meralion-3-asr" and ei.normalise("nvidia-ml-py") == "nvidia-ml-py"


def test_the_same_environment_on_another_machine_passes_even_when_paths_python_patch_version_freeze_text_and_pip_differ():
    result = check()
    assert result["passed"] is True and result["differences"] == [] and "freeze text" in result["never_compared"] and "paths" in result["never_compared"]


def informational(**changes):
    here = current()
    here["informational"] = {**here["informational"], **changes}
    return here


def test_codexs_probe_python_3_11_on_windows_is_no_longer_accepted():
    """Python 3.11.9 on Windows/AMD64 used to pass: recorded python, platform and machine were never looked at."""
    result = check(informational(python="3.11.9", platform="Windows-11", machine="AMD64", system="Windows"))
    assert result["passed"] is False
    text = " ".join(result["differences"])
    assert "operating system is 'Windows'" in text and "machine architecture is 'AMD64'" in text and "Python is '3.11.9'" in text and "not the same minor version" in text


@pytest.mark.parametrize("changes, fragment", [
    (dict(system="Darwin"), "operating system"), (dict(system=None), "operating system"), (dict(machine="aarch64"), "machine architecture"), (dict(machine="AMD64"), "machine architecture"),
    (dict(python="3.11.9"), "Python is '3.11.9'"), (dict(python="3.13.1"), "Python is '3.13.1'"), (dict(python=None), "Python is None"),
])
def test_the_deployment_target_is_enforced_one_property_at_a_time(changes, fragment):
    result = check(informational(**changes))
    assert result["passed"] is False and any(fragment in d for d in result["differences"]), result["differences"]


def test_a_python_patch_version_difference_stays_allowed_and_the_reference_minor_must_match():
    assert check(informational(python="3.12.3"))["passed"] is True and check(informational(python="3.12.99"))["passed"] is True
    other_reference = reference()
    other_reference["informational"]["python"] = "3.11.9"
    result = check(informational(python="3.12.3"), ref=other_reference)
    assert result["passed"] is False and any("not the same minor version" in d for d in result["differences"])


def test_a_record_stores_the_system_and_the_machine():
    recorded = ei.record()
    assert recorded["informational"]["system"] and recorded["informational"]["machine"] and recorded["informational"]["python"]


def test_each_real_difference_is_named():
    cases = [
        (dict(pins={**current()["pins"], "numpy": "2.3.0"}), "package numpy: reference 2.2.6, here 2.3.0"),
        (dict(pins={k: v for k, v in current()["pins"].items() if k != "numpy"}), "missing package numpy"),
        (dict(pins={**current()["pins"], "extra-lib": "1.0"}), "extra package extra-lib 1.0"),
        (dict(meralion_artifact_sha256="f" * 64), "wheel hash"),
        (dict(meralion_artifact_sha256=None), "wheel hash"),
        (dict(torch={"version": "2.9.1+cu126", "cuda": "12.6"}), "torch version"),
        (dict(torch={"version": "2.9.1+cu128", "cuda": "12.6"}), "torch cuda"),
        (dict(audio={**reference()["audio"], "libsndfile": "1.2.0"}), "audio library libsndfile"),
        (dict(audio={**reference()["audio"], "soxr": "1.0.0"}), "audio library soxr"),
    ]
    for override, fragment in cases:
        result = check(current(**override))
        assert result["passed"] is False and any(fragment in d for d in result["differences"]), (override, result["differences"])


def test_the_wheel_hash_is_checked_against_the_verified_manifest_too_and_a_manifest_without_one_fails():
    other = "e" * 64
    result = check(current(meralion_artifact_sha256=other), reference(meralion_artifact_sha256=other))
    assert result["passed"] is False and any("not the verified package manifest's" in d for d in result["differences"])
    assert any("names no artifact sha256" in d for d in check(manifest={"artifact": {}})["differences"])


@pytest.mark.parametrize("states, expect, ok", [
    (["patched"], "patched", True), (["original"], "patched", False), ([], "patched", False), (["original"], "original", True), (["not_installed"], "original", True), (["patched"], "original", False),
    (["unknown"], "none", False), (["original"], "none", True), ([], "none", True), (["patched"], "none", False)])
def test_the_tracked_patch_state_must_match_the_expectation(states, expect, ok):
    here = current(patches=[{"target": "vllm/platforms/cuda.py", "state": state, "installed_sha256": "x"} for state in states])
    assert check(here, expect=expect)["passed"] is ok


def test_a_failing_verify_environment_fails_even_when_every_pin_matches():
    result = check(verify=3)
    assert result["passed"] is False and any("verify-environment exited 3" in d for d in result["differences"]) and any("identical pins are not enough" in d for d in result["differences"])


def test_a_raw_freeze_text_difference_alone_never_fails():
    here = current()
    here["informational"]["freeze_text_sha256"] = "9" * 64
    assert check(here)["passed"] is True


def test_the_wheel_hash_is_read_from_direct_url_json_without_a_path(tmp_path, monkeypatch):
    dist = tmp_path / "meralion_3_asr-0.1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: meralion-3-asr\nVersion: 0.1.0\n", encoding="utf-8")
    (dist / "direct_url.json").write_text(json.dumps({"url": "file:///some/where/else/meralion_3_asr-0.1.0-py3-none-any.whl", "archive_info": {"hash": f"sha256={WHEEL}"}}), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert ei.artifact_hash_of() == WHEEL
    (dist / "direct_url.json").write_text(json.dumps({"url": "file:///x", "archive_info": {"hashes": {"sha256": "c" * 64}}}), encoding="utf-8")
    assert ei.artifact_hash_of() == "c" * 64
    (dist / "direct_url.json").unlink()
    assert ei.artifact_hash_of() is None
    assert ei.artifact_hash_of("definitely-not-installed") is None


def test_the_deployment_target_constants_are_linux_x86_64_and_python_3_12():
    assert (ei.TARGET_SYSTEM, ei.TARGET_MACHINE, ei.TARGET_PYTHON_MINOR) == ("Linux", "x86_64", "3.12")


def test_record_and_check_round_trip_in_this_environment(tmp_path, monkeypatch):
    import platform
    monkeypatch.setattr(ei, "TARGET_SYSTEM", platform.system())                       # this test checks the mechanics on whatever machine runs it; the real target is pinned by the test above
    monkeypatch.setattr(ei, "TARGET_MACHINE", platform.machine())
    monkeypatch.setattr(ei, "TARGET_PYTHON_MINOR", ".".join(platform.python_version().split(".")[:2]))
    monkeypatch.setattr(ei, "artifact_hash_of", lambda package=ei.PACKAGE: WHEEL)
    freeze = tmp_path / "freeze.txt"
    freeze.write_text("meralion-3-asr @ file:///a/b.whl#sha256=abc\n", encoding="utf-8")
    ref = tmp_path / "reference.json"
    assert ei.main(["--record", str(ref), "--freeze-file", str(freeze)]) == 0
    recorded = json.loads(ref.read_text(encoding="utf-8"))
    assert recorded["schema"] == "environment-reference-v1" and "pytest" in recorded["pins"] and recorded["informational"]["freeze_text_sha256"] == hashlib.sha256(freeze.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST), encoding="utf-8")
    out = tmp_path / "check.json"
    assert ei.main(["--check", str(ref), "--package-manifest", str(manifest), "--verify-environment-exit", "0", "--expect-patches", "original", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True
    assert ei.main(["--check", str(ref), "--package-manifest", str(manifest), "--verify-environment-exit", "2", "--expect-patches", "original", "--out", str(out)]) == 1
    recorded["pins"]["pytest"] = "0.0.1"
    ref.write_text(json.dumps(recorded), encoding="utf-8")
    assert ei.main(["--check", str(ref), "--package-manifest", str(manifest), "--verify-environment-exit", "0", "--expect-patches", "original", "--out", str(out)]) == 1


def test_the_tool_uses_the_standard_library_only():
    import ast
    tree = ast.parse((TOOLS / "env_identity.py").read_text(encoding="utf-8"))
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= {"argparse", "hashlib", "json", "platform", "re", "sys", "importlib", "pathlib", "torch", "soundfile", "meralion_worker"}       # torch, soundfile and the worker's identity only inside guarded functions
    top_level = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names} | {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
    assert not top_level & {"torch", "soundfile", "meralion_worker"}


# ---- audio_fingerprint -----------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Rec:
    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype=np.float32)
        self.num_samples = len(self.samples)

    def get(self, a, b):
        return self.samples[a:b]


def test_the_fingerprint_is_the_count_and_the_hash_of_the_loaded_float32_array():
    rec = Rec([0.0, 0.5, -0.5])
    assert af.fingerprint_recording(rec) == {"num_samples": 3, "sha256": hashlib.sha256(np.asarray([0.0, 0.5, -0.5], dtype=np.float32).tobytes()).hexdigest()}
    assert af.fingerprint_recording(Rec([0.0, 0.5, -0.25]))["sha256"] != af.fingerprint_recording(rec)["sha256"]


def test_fingerprinting_selects_calls_by_name_and_refuses_unknown_ones(tmp_path):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"job_name": "A", "input": "a.wav"}, {"job_name": "B", "input": "b.wav"}]}), encoding="utf-8")
    loaded = {"a.wav": Rec([1, 2, 3]), "b.wav": Rec([4, 5])}
    found = af.fingerprint_jobs(jobs, ["B"], loader=lambda call, path: loaded[path])
    assert list(found) == ["B"] and found["B"]["num_samples"] == 2
    assert list(af.fingerprint_jobs(jobs, None, loader=lambda call, path: loaded[path])) == ["A", "B"]
    with pytest.raises(SystemExit, match="not in the jobs file"):
        af.fingerprint_jobs(jobs, ["C"], loader=lambda call, path: loaded[path])


def test_a_different_value_is_a_difference_and_a_missing_recording_is_one_too():
    ref = {"A": {"num_samples": 3, "sha256": "1"}, "B": {"num_samples": 2, "sha256": "2"}}
    assert af.compare(ref, dict(ref)) == (True, [])
    passed, differences = af.compare(ref, {"A": {"num_samples": 3, "sha256": "9"}, "C": {"num_samples": 1, "sha256": "3"}})
    assert passed is False and any(d.startswith("A:") for d in differences) and any("B: not fingerprinted here" in d for d in differences) and any("C: not in the reference" in d for d in differences)


def test_the_command_line_writes_the_fingerprint_and_returns_one_on_a_mismatch(tmp_path, monkeypatch):
    from meralion_worker import audio
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"job_name": "A", "input": "a.wav"}]}), encoding="utf-8")
    monkeypatch.setattr(audio, "load_recording", lambda call, path: Rec([1, 2, 3]))
    out = tmp_path / "fp.json"
    assert af.main(["--jobs", str(jobs), "--out", str(out)]) == 0
    reference = json.loads(out.read_text(encoding="utf-8"))
    assert reference["schema"] == "audio-fingerprint-v1" and "recordings" in reference and "libraries" in reference
    check = tmp_path / "check.json"
    assert af.main(["--jobs", str(jobs), "--check", str(out), "--out", str(check)]) == 0 and json.loads(check.read_text(encoding="utf-8"))["passed"] is True
    monkeypatch.setattr(audio, "load_recording", lambda call, path: Rec([1, 2, 4]))
    assert af.main(["--jobs", str(jobs), "--check", str(out), "--out", str(check)]) == 1 and json.loads(check.read_text(encoding="utf-8"))["passed"] is False
