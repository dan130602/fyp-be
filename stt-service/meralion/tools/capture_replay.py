"""CPU replay of a live capture of the vLLM adapter (Stage 3 follow-up E, section 8.5). Verification only; never part of the image; needs numpy and the verified wheel, not torch or vllm.

    python tools/capture_replay.py --capture capture.json --wheel artifacts/meralion_3_asr-0.1.0-py3-none-any.whl \
        --package-manifest configs/third_party/meralion-3-asr-0.1.0.json --out capture_replay.json

The capture (`vllm_windows.py --capture`) is ONE real `llm.generate` call made by the adapter: the request structure and audio hashes, the sampling object passed, the completions, the intended piece
hashes and the installed package file hashes. This tool checks, each as a named check:

  wheel_sha256_equals_manifest_artifact      the wheel given here is the verified artifact of the package manifest
  installed_files_equal_verified_wheel       the package files the live run used (their hashes in the capture) are those of the verified wheel
  live_request_correspondence                every captured request audio hash equals the hash of the intended piece (one request per intended piece, one audio-file hash per call)
  replay_requests_equal                      the real `VllmBackend.transcribe_chunks` of the wheel and `VllmDecoder.decode_many` build equal requests from the same pieces
  replay_sampling_equal                      by EFFECTIVE PARAMETER VALUES (never object identity between a 1.0 and a 1.05 object): what the adapter passed, what an isolated package replay
                                             instance configured with the profile's effective values passes, the captured object and the frozen table agree; the live backend's original object
                                             was not mutated (its attributes before and after the capture are the stock normalised values) and the package's REQUESTED configuration is the stock request
  replay_text_equal                          the package's and the adapter's cleaned texts are equal on every captured completion

Synthetic pieces (deterministic noise of the captured shapes) establish request construction, parameter passing and the text cleanup on real completions; the correspondence between the live
requests and the intended audio rests ONLY on the hashes recorded in the live capture (`live_request_correspondence`). The report says so.
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys
import types
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meralion_worker.decoders import frozen                                                  # noqa: E402
from meralion_worker.decoders import vllm_decoder as vd                                       # noqa: E402
from meralion_worker.decoders.base import DecodeRequest                                       # noqa: E402

STOCK_ID = "stock-vllm-package-v1"
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "max_tokens", "repetition_penalty")
WHEEL_MEMBERS = {"backends/vllm_backend.py": "meralion_3_asr/backends/vllm_backend.py", "prompts.py": "meralion_3_asr/prompts.py"}
ENVIRONMENT = ("MERALION_NGRAM_SIZE", "VLLM_ENABLE_V1_MULTIPROCESSING", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
EPS = 1e-5


class ReplaySamplingParams:
    """Stand-in for vllm.SamplingParams with the normalisation of vLLM 0.16.0 (sampling_params.py:367-372): zero temperature means greedy, which resets top_p, top_k and min_p."""

    def __init__(self, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=16, repetition_penalty=1.0, stop=None, stop_token_ids=None):
        self.temperature, self.top_p, self.top_k, self.min_p = temperature, top_p, top_k, min_p
        self.max_tokens, self.repetition_penalty = max_tokens, repetition_penalty
        self.stop = [] if stop is None else stop
        self.stop_token_ids = [] if stop_token_ids is None else stop_token_ids
        if self.temperature < EPS:
            self.top_p, self.top_k, self.min_p = 1.0, 0, 0.0

    def __repr__(self):
        return "ReplaySamplingParams(" + ", ".join(f"{k}={getattr(self, k)!r}" for k in (*SAMPLING_KEYS, "stop", "stop_token_ids")) + ")"


class ScriptedLLM:
    """Records every generate call and answers with the captured completions, in request order."""

    def __init__(self, completions):
        self.completions = completions
        self.calls = []

    def generate(self, requests, sampling_params):
        self.calls.append((requests, sampling_params))
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(**c)]) for c in self.completions[:len(requests)]]


@contextlib.contextmanager
def real_backend_class(wheel):
    """The REAL VllmBackend class of the verified wheel, imported from the zip with a stub librosa (the package imports it at load; transcribe_chunks never uses it). Restores sys.path and sys.modules."""
    before_modules, before_path = set(sys.modules), list(sys.path)
    sys.path.insert(0, str(wheel))
    if "librosa" not in sys.modules:
        try:
            import librosa                                                                    # noqa: F401
        except ImportError:
            sys.modules["librosa"] = types.ModuleType("librosa")
    try:
        from meralion_3_asr.backends.vllm_backend import VllmBackend
        yield VllmBackend
    finally:
        for name in set(sys.modules) - before_modules:
            if name.startswith("meralion_3_asr") or name == "librosa":
                del sys.modules[name]
        sys.path[:] = before_path


@contextlib.contextmanager
def restored_environment():
    saved = {key: os.environ.get(key) for key in ENVIRONMENT}
    os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)                                    # the replay builds decoders of any profile: the launcher's own value must not decide
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attrs_of(sampling):
    return {key: getattr(sampling, key, None) for key in (*SAMPLING_KEYS, "stop", "stop_token_ids")}


def view_subset(attrs, keys=SAMPLING_KEYS):
    return {key: attrs.get(key) for key in keys}


def synthetic_pieces(rows):
    """Deterministic float32 noise of the captured shapes (no audio is stored in a capture and none is invented from it)."""
    rng = np.random.default_rng(20261116)
    return [rng.standard_normal(tuple(row["audio_shape"])).astype(np.float32) * 0.1 for row in rows]


def requests_equal(left, right):
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if set(a) != set(b) or a["prompt"] != b["prompt"] or set(a["multi_modal_data"]) != set(b["multi_modal_data"]):
            return False
        (audio_a, rate_a), (audio_b, rate_b) = a["multi_modal_data"]["audio"], b["multi_modal_data"]["audio"]
        if rate_a != rate_b or audio_a.dtype != audio_b.dtype or not np.array_equal(audio_a, audio_b):
            return False
    return True


def replay(capture, wheel):
    """-> {check name: bool} for the replay checks, plus details."""
    settings_id = capture["settings_id"]
    rows, completions = capture["call"]["requests"], capture["call"]["completions"]
    pieces = synthetic_pieces(rows)
    table = frozen.frozen_settings(settings_id)
    stock = frozen.frozen_settings(STOCK_ID)
    effective = ReplaySamplingParams(**table["requested_sampling"])                       # what the profile's frozen table says the adapter must pass
    package_original = ReplaySamplingParams(**stock["requested_sampling"])                # the package's own object, built from the package's request
    original_snapshot = attrs_of(package_original)
    results = {}
    with real_backend_class(wheel) as backend_class, restored_environment():
        package_llm = ScriptedLLM(completions)
        isolated = object.__new__(backend_class)                                            # an ISOLATED package instance (never the adapter's backend)
        isolated.llm, isolated.sampling_params, isolated._prompt = package_llm, effective, capture["prompt"]
        package_texts = backend_class.transcribe_chunks(isolated, pieces)
        adapter_llm = ScriptedLLM(completions)
        client_name = (capture.get("execution") or {}).get("engine_core_client")                  # the engine client class the live decoder OBSERVED, mirrored so the decoder's own check runs
        if client_name:
            adapter_llm.llm_engine = types.SimpleNamespace(engine_core=type(client_name, (), {})())
        adapter_backend = types.SimpleNamespace(llm=adapter_llm, sampling_params=package_original, _prompt=capture["prompt"])
        decoder = vd.VllmDecoder(backend=adapter_backend, settings_id=settings_id,
                                 package_info=lambda: {"requested_sampling": dict(stock["requested_sampling"]), "constructor_defaults": dict(table["constructor_defaults"])})
        decoded = decoder.decode_many([DecodeRequest("replay", 0, len(piece), piece) for piece in pieces])
    package_requests, package_params = package_llm.calls[0]
    adapter_requests, adapter_params = adapter_llm.calls[0]
    expected_requests = [{"prompt": capture["prompt"], "multi_modal_data": {"audio": (piece, 16000)}} for piece in pieces]
    results["replay_requests_equal"] = requests_equal(package_requests, adapter_requests) and requests_equal(adapter_requests, expected_requests) and \
        all(r["audio_rate"] == 16000 and r["audio_dtype"] == "float32" and r["prompt"] == capture["prompt"] for r in rows)
    normalised = view_subset(table["normalised_sampling"], SAMPLING_KEYS)
    live = capture["call"]["sampling"]["attrs"]
    results["replay_sampling_equal"] = all((
        view_subset(attrs_of(adapter_params)) == view_subset(attrs_of(package_params)) == view_subset(live) == normalised,           # effective values agree everywhere and equal the frozen table
        not live.get("stop") and not live.get("stop_token_ids"),                                                                    # no stop configured live
        adapter_backend.sampling_params is package_original and attrs_of(package_original) == original_snapshot,                    # the adapter never touched the package's object
        (adapter_params is package_original) == (settings_id in (STOCK_ID, frozen.STOCK_VLLM_SCHED0_ID)),                           # stock generates with the package's own object, a candidate with its own
        view_subset(capture["package_original"]["attrs"]) == view_subset(stock["normalised_sampling"], SAMPLING_KEYS),             # the live package object (normalised: top_k 50 became 0) is the stock one
        capture["package_original_after"]["attrs"] == capture["package_original"]["attrs"],                                           # and it was not mutated by the run
        capture["package_requested"] == stock["requested_sampling"],                                                                # the package's REQUESTED configuration is the stock request (top_k 50)
    ))
    results["replay_text_equal"] = [r.text for r in decoded] == package_texts == [vd.clean_text(c["text"]) for c in completions]
    return results


def check_capture(capture, wheel, manifest):
    checks = {}
    artifact = (manifest.get("artifact") or {}).get("sha256")
    checks["wheel_sha256_equals_manifest_artifact"] = bool(artifact) and sha256_file(wheel) == artifact
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        wheel_hashes = {name: hashlib.sha256(archive.read(member)).hexdigest() if member in names else None for name, member in WHEEL_MEMBERS.items()}
    package = capture.get("package") or {}
    checks["installed_files_equal_verified_wheel"] = package.get("files_sha256") == wheel_hashes and package.get("version") == manifest.get("version")
    requests, intended = capture["call"]["requests"], capture["intended"]
    calls = {item["call"] for item in intended}
    checks["live_request_correspondence"] = bool(requests) and capture.get("generate_calls_captured") == 1 and len(requests) == len(intended) and \
        [r["audio_sha256"] for r in requests] == [i["piece_sha256"] for i in intended] and capture.get("live_request_correspondence") is True and \
        calls == set(capture.get("audio_file_sha256") or {}) and all(len(h) == 64 for h in (capture.get("audio_file_sha256") or {}).values())
    try:
        checks.update(replay(capture, wheel))
    except Exception as error:                                                                # noqa: BLE001  a replay that cannot run is a failed check, with its reason recorded
        checks.update({name: False for name in ("replay_requests_equal", "replay_sampling_equal", "replay_text_equal")})
        checks["replay_error"] = False
        print(f"capture_replay: the replay could not run: {type(error).__name__}: {error}", file=sys.stderr)
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"report already exists, not overwriting it: {args.out}")
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    manifest = json.loads(args.package_manifest.read_text(encoding="utf-8"))
    checks = check_capture(capture, args.wheel, manifest)
    report = {"tool": "capture_replay", "checks": checks, "passed": bool(checks) and all(checks.values()),
              "evidence": {"capture_sha256": sha256_file(args.capture), "wheel_sha256": sha256_file(args.wheel), "package_manifest_sha256": sha256_file(args.package_manifest),
                           "worker_commit": capture.get("worker_git_commit"), "settings_id": capture.get("settings_id"), "execution_kind": (capture.get("execution") or {}).get("mode")},
              "scope": "Synthetic pieces of the captured shapes establish request construction, sampling-parameter passing and the text cleanup on REAL completions; the correspondence between the live "
                       "requests and the intended audio rests only on the hashes recorded in the live capture (live_request_correspondence)."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": checks}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
