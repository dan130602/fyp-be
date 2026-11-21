"""tools/capture_replay.py (Stage 3 follow-up E, section 8.5): the CPU replay of a live capture of the adapter, checked against the REAL package code of the verified wheel.
Synthetic captures only; skipped where the git-ignored wheel is absent."""
import copy
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from meralion_worker import config as cfg
from meralion_worker.decoders import frozen

ROOT = Path(__file__).resolve().parent.parent
WHEEL = ROOT / "artifacts" / "meralion_3_asr-0.1.0-py3-none-any.whl"
MANIFEST_PATH = cfg.CONFIG_ROOT / "third_party" / "meralion-3-asr-0.1.0.json"
pytestmark = pytest.mark.skipif(not WHEEL.is_file(), reason="artifacts/meralion_3_asr-0.1.0-py3-none-any.whl is not present")

STOCK, CANDIDATE, STOCK_SCHED0, CANDIDATE_SCHED0 = "stock-vllm-package-v1", "vllm-rp105-v1", "stock-vllm-package-sched0-v1", "vllm-rp105-sched0-v1"
IDS = (STOCK, CANDIDATE, STOCK_SCHED0, CANDIDATE_SCHED0)
TEXTS = ["hello", "<Speaker1>: hello world", "  Speaker 1:  spaced  ", "好的，谢谢您。", "<Speaker1>hello", ""]
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def load_tool():
    spec = importlib.util.spec_from_file_location("capture_replay", ROOT / "tools" / "capture_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay_tool = load_tool()
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def piece_hash(piece):
    return hashlib.sha256(np.ascontiguousarray(piece, dtype=np.float32).tobytes()).hexdigest()


def wheel_hashes():
    with zipfile.ZipFile(WHEEL) as archive:
        return {name: hashlib.sha256(archive.read(member)).hexdigest() for name, member in replay_tool.WHEEL_MEMBERS.items()}


def view(attrs):
    return {"repr": "SamplingParams(...)", "attrs": dict(attrs)}


def make_capture(settings_id=CANDIDATE, sizes=(16000, 24000, 8000, 480000, 100, 16000)):
    """A capture as `vllm_windows.py --capture` would write it from a faithful live run."""
    table = frozen.frozen_settings(settings_id)
    stock = frozen.frozen_settings(STOCK)
    rng = np.random.default_rng(1)
    pieces = [rng.standard_normal(size).astype(np.float32) * 0.1 for size in sizes]
    live = {**table["normalised_sampling"], "stop": [], "stop_token_ids": []}
    original = {**stock["normalised_sampling"], "stop": [], "stop_token_ids": []}
    intended = [{"label": f"w{i}", "call": "CALL_A" if i % 2 == 0 else "CALL_B", "start": 0, "end": len(p), "piece_sha256": piece_hash(p)} for i, p in enumerate(pieces)]
    return {"format": 1, "tool": "vllm_windows", "settings_id": settings_id, "worker_git_commit": COMMIT, "prompt": "THE PROMPT",
            "execution": ({"mode": "default", "variable": None, "engine_core_client": "SyncMPClient"} if frozen.execution_of(settings_id) is None
                          else {"mode": "profile-in-process", "variable": "0", "engine_core_client": "InprocClient"}),
            "package": {"version": MANIFEST["version"], "files_sha256": wheel_hashes()}, "package_requested": dict(stock["requested_sampling"]),
            "package_original": view(original), "package_original_after": view(original), "intended": intended, "audio_file_sha256": {"CALL_A": "a" * 64, "CALL_B": "b" * 64},
            "generate_calls_captured": 1, "live_request_correspondence": True,
            "call": {"requests": [{"prompt": "THE PROMPT", "audio_rate": 16000, "audio_dtype": "float32", "audio_shape": [len(p)], "audio_sha256": piece_hash(p)} for p in pieces],
                     "sampling": view(live),
                     "completions": [{"text": TEXTS[i % len(TEXTS)], "token_ids": [1, 2, 3], "finish_reason": "stop", "stop_reason": None} for i in range(len(pieces))]}}


def check(capture):
    return replay_tool.check_capture(capture, WHEEL, MANIFEST)


@pytest.mark.parametrize("settings_id", IDS)
def test_a_faithful_capture_of_every_profile_passes_every_check(settings_id):
    checks = check(make_capture(settings_id))
    assert checks and all(checks.values()), checks
    assert set(checks) == {"wheel_sha256_equals_manifest_artifact", "installed_files_equal_verified_wheel", "live_request_correspondence", "replay_requests_equal", "replay_sampling_equal",
                           "replay_text_equal"}


def test_the_replay_leaves_no_trace_in_the_environment_or_the_import_state():
    import os
    before_modules, before_path, before_env = set(sys.modules), list(sys.path), dict(os.environ)
    check(make_capture(CANDIDATE_SCHED0))                                                    # a scheduling profile sets the variable inside the decoder: it must not leak
    assert dict(os.environ) == before_env and sys.path == before_path
    assert not [name for name in set(sys.modules) - before_modules if name.startswith("meralion_3_asr")]


# ---- sampling: effective values, never object identity ---------------------------------------------------------------------------------------------------------------------------

def test_a_candidate_capture_whose_passed_object_lacks_the_override_fails_the_sampling_check():
    capture = make_capture(CANDIDATE)
    capture["call"]["sampling"]["attrs"]["repetition_penalty"] = 1.0
    checks = check(capture)
    assert checks["replay_sampling_equal"] is False and checks["replay_requests_equal"] is True


def test_a_stock_capture_carrying_the_candidates_penalty_fails_and_so_does_a_penalty_other_than_the_profiles():
    for settings_id, wrong in ((STOCK, 1.05), (STOCK_SCHED0, 1.05), (CANDIDATE, 1.1), (CANDIDATE_SCHED0, 1.0)):
        capture = make_capture(settings_id)
        capture["call"]["sampling"]["attrs"]["repetition_penalty"] = wrong
        assert check(capture)["replay_sampling_equal"] is False, (settings_id, wrong)


def test_a_mutated_original_package_object_fails():
    capture = make_capture(CANDIDATE)
    capture["package_original_after"]["attrs"]["repetition_penalty"] = 1.05                  # the package's own object was assigned the candidate's value during the run
    assert check(capture)["replay_sampling_equal"] is False
    capture = make_capture(CANDIDATE)
    capture["package_original"]["attrs"]["repetition_penalty"] = 1.05                        # or it was already not the stock object when the run began
    assert check(capture)["replay_sampling_equal"] is False


def test_the_original_object_is_compared_normalised_and_the_request_separately():
    capture = make_capture(STOCK)
    assert capture["package_original"]["attrs"]["top_k"] == 0 and capture["package_requested"]["top_k"] == 50 and check(capture)["replay_sampling_equal"] is True
    capture["package_original"]["attrs"]["top_k"] = 50                                       # a requested value in the normalised object is wrong: greedy sampling resets it
    assert check(capture)["replay_sampling_equal"] is False
    capture = make_capture(STOCK)
    capture["package_requested"]["top_k"] = 0                                                # the package's request must stay what it shipped
    assert check(capture)["replay_sampling_equal"] is False


def test_a_configured_stop_in_the_live_object_fails():
    capture = make_capture(STOCK)
    capture["call"]["sampling"]["attrs"]["stop"] = ["###"]
    assert check(capture)["replay_sampling_equal"] is False


def test_the_replay_never_uses_object_identity_between_a_stock_and_a_candidate_object(monkeypatch):
    seen = {}
    original = replay_tool.ReplaySamplingParams.__init__

    def spy(self, *args, **kwargs):
        seen.setdefault("repetition_penalties", []).append(kwargs.get("repetition_penalty"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(replay_tool.ReplaySamplingParams, "__init__", spy)
    assert all(check(make_capture(CANDIDATE)).values())
    assert set(seen["repetition_penalties"]) >= {1.0, 1.05}                                  # the package's 1.0 object and the candidate's 1.05 objects coexist and are compared by value


# ---- requests and text ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def test_a_profile_capture_that_observed_the_wrong_engine_client_fails_the_replay():
    capture = make_capture(CANDIDATE_SCHED0)
    capture["execution"]["engine_core_client"] = "SyncMPClient"                              # the live decoder would have refused; a capture that says so is not a faithful one
    assert check(capture)["replay_sampling_equal"] is False and check(capture)["replay_text_equal"] is False
    capture = make_capture(CANDIDATE)
    capture["execution"]["engine_core_client"] = "InprocClient"
    assert check(capture)["replay_requests_equal"] is False


def test_a_request_hash_that_is_not_the_intended_pieces_fails_the_live_correspondence():
    capture = make_capture(CANDIDATE)
    capture["call"]["requests"][2]["audio_sha256"] = "0" * 64
    assert check(capture)["live_request_correspondence"] is False
    capture = make_capture(CANDIDATE)
    capture["live_request_correspondence"] = False
    assert check(capture)["live_request_correspondence"] is False
    capture = make_capture(CANDIDATE)
    capture["intended"].pop()
    assert check(capture)["live_request_correspondence"] is False
    capture = make_capture(CANDIDATE)
    capture["audio_file_sha256"].pop("CALL_B")
    assert check(capture)["live_request_correspondence"] is False
    capture = make_capture(CANDIDATE)
    capture["generate_calls_captured"] = 2
    assert check(capture)["live_request_correspondence"] is False


def test_a_prompt_or_rate_or_dtype_that_differs_from_the_captured_request_fails_the_request_check():
    for field, value in (("audio_rate", 8000), ("audio_dtype", "float64"), ("prompt", "ANOTHER")):
        capture = make_capture(CANDIDATE)
        capture["call"]["requests"][0][field] = value
        assert check(capture)["replay_requests_equal"] is False, field


def test_the_text_check_runs_on_the_captured_completions_through_both_code_paths():
    capture = make_capture(CANDIDATE)
    assert check(capture)["replay_text_equal"] is True
    capture["call"]["completions"][1]["text"] = "<Speaker1>: <Speaker1>: twice"             # the real package strips the tags in sequence on the trimmed text
    assert check(capture)["replay_text_equal"] is True


def test_an_adapter_whose_cleanup_differs_from_the_packages_fails_the_text_check(monkeypatch):
    from meralion_worker.decoders import vllm_decoder
    monkeypatch.setattr(vllm_decoder, "SPEAKER_TAGS", ())                                    # the adapter no longer strips the speaker tags the package strips
    checks = check(make_capture(CANDIDATE))
    assert checks["replay_text_equal"] is False and checks["replay_requests_equal"] is True


def test_an_adapter_that_sends_different_audio_than_the_package_fails_the_request_check(monkeypatch):
    from meralion_worker.decoders import vllm_decoder
    original = vllm_decoder.build_requests
    monkeypatch.setattr(vllm_decoder, "build_requests", lambda prompt, pieces: original(prompt, [piece * np.float32(0.5) for piece in pieces]))
    checks = check(make_capture(CANDIDATE))
    assert checks["replay_requests_equal"] is False and checks["replay_text_equal"] is True


@pytest.mark.parametrize("settings_id", (STOCK, STOCK_SCHED0))
def test_a_stock_adapter_that_stops_using_the_packages_own_sampling_object_fails(monkeypatch, settings_id):
    from meralion_worker.decoders import vllm_decoder
    import copy
    original_init = vllm_decoder.VllmDecoder.__init__

    def copying(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.sampling_params = copy.copy(self.sampling_params)                               # same values, but no longer the package's own object

    monkeypatch.setattr(vllm_decoder.VllmDecoder, "__init__", copying)
    assert check(make_capture(settings_id))["replay_sampling_equal"] is False


# ---- the package and the wheel ---------------------------------------------------------------------------------------------------------------------------------------------------

def test_installed_files_that_are_not_the_verified_wheels_fail():
    capture = make_capture(CANDIDATE)
    capture["package"]["files_sha256"]["backends/vllm_backend.py"] = "f" * 64
    assert check(capture)["installed_files_equal_verified_wheel"] is False
    capture = make_capture(CANDIDATE)
    capture["package"]["version"] = "0.1.1"
    assert check(capture)["installed_files_equal_verified_wheel"] is False
    capture = make_capture(CANDIDATE)
    capture["package"]["files_sha256"].pop("prompts.py")
    assert check(capture)["installed_files_equal_verified_wheel"] is False


def test_a_wheel_that_is_not_the_manifests_artifact_fails(tmp_path):
    other = tmp_path / "other.whl"
    with zipfile.ZipFile(WHEEL) as source, zipfile.ZipFile(other, "w") as target:
        for item in source.infolist():
            target.writestr(item, source.read(item) + (b"\n# changed" if item.filename.endswith("prompts.py") else b""))
    checks = replay_tool.check_capture(make_capture(CANDIDATE), other, MANIFEST)
    assert checks["wheel_sha256_equals_manifest_artifact"] is False and checks["installed_files_equal_verified_wheel"] is False


def test_a_replay_that_cannot_run_is_a_failed_check_not_a_crash(tmp_path, capsys):
    broken = tmp_path / "broken.whl"
    with zipfile.ZipFile(broken, "w") as archive:
        archive.writestr("meralion_3_asr/__init__.py", "")
    checks = replay_tool.check_capture(make_capture(CANDIDATE), broken, MANIFEST)
    assert checks["replay_requests_equal"] is False and checks["replay_sampling_equal"] is False and checks["replay_text_equal"] is False
    assert "the replay could not run" in capsys.readouterr().err


# ---- the report and the command ------------------------------------------------------------------------------------------------------------------------------------------------------

def test_the_report_binds_the_capture_the_wheel_the_manifest_the_commit_and_the_profile(tmp_path):
    capture_path, out = tmp_path / "capture.json", tmp_path / "report.json"
    capture_path.write_text(json.dumps(make_capture(CANDIDATE_SCHED0)), encoding="utf-8")
    argv = ["--capture", str(capture_path), "--wheel", str(WHEEL), "--package-manifest", str(MANIFEST_PATH), "--out", str(out)]
    assert replay_tool.main(argv) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["passed"] is True and "synthetic" in report["scope"].lower() and "hashes recorded in the live capture" in report["scope"]
    assert report["evidence"] == {"capture_sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(), "wheel_sha256": MANIFEST["artifact"]["sha256"],
                                  "package_manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(), "worker_commit": COMMIT, "settings_id": CANDIDATE_SCHED0,
                                  "execution_kind": "profile-in-process"}
    with pytest.raises(SystemExit, match="already exists"):
        replay_tool.main(argv)


def test_a_failing_capture_gives_exit_code_one_and_a_failed_report(tmp_path):
    capture = make_capture(CANDIDATE)
    capture["call"]["sampling"]["attrs"]["repetition_penalty"] = 1.0
    capture_path, out = tmp_path / "capture.json", tmp_path / "report.json"
    capture_path.write_text(json.dumps(capture), encoding="utf-8")
    assert replay_tool.main(["--capture", str(capture_path), "--wheel", str(WHEEL), "--package-manifest", str(MANIFEST_PATH), "--out", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is False


def test_the_wheel_loader_and_the_adapter_test_load_the_identical_class_file():
    with replay_tool.real_backend_class(WHEEL) as backend_class:
        import inspect
        loaded = Path(inspect.getsourcefile(backend_class)) if inspect.getsourcefile(backend_class) else None
        source = inspect.getsource(backend_class.transcribe_chunks)
    with zipfile.ZipFile(WHEEL) as archive:
        assert hashlib.sha256(archive.read("meralion_3_asr/backends/vllm_backend.py")).hexdigest() == wheel_hashes()["backends/vllm_backend.py"]
    assert "multi_modal_data" in source and loaded is not None
