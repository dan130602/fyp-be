"""Decode the named problem windows through the worker's vLLM adapter in the NORMAL execution mode, in repeated passes (Stage 3 follow-up D; never part of the image).

    python vllm_windows.py --settings-id vllm-rp105-v1 --jobs jobs.json --windows problem_windows.json --passes batch,batch,single --out windows.json [--package-parity]

One process, one engine, the settings id's own execution profile (the LAUNCHER must not set VLLM_ENABLE_V1_MULTIPROCESSING=0: a default profile runs multiprocess, a scheduling profile sets the
variable itself in the decoder; the diagnostic probe is a separate process). Each pass decodes every window either as ONE batch ("batch") or one
window at a time ("single"). For every window and pass it records the text, the mapped stop reason, the generated token count and the sha256 of the generated token ids. Differences between passes are REPORTED; no
seed, enforce-eager or other determinism setting is changed. With --package-parity the harness gives the package's own `transcribe_chunks` the adapter's EFFECTIVE sampling object (harness-only assignment on the
package instance, never in the worker) and compares the texts, and checks the live D6 (normalised sampling) and D7 (finish reasons) facts.

--capture PATH (follow-up E; harness only, the adapter and the worker are untouched) records, for the FIRST adapter batch (pass 0, which must be a batch pass), what the adapter sent to
`llm.generate` and what it got back: per request the prompt, audio rate, dtype, shape and sha256 (never the audio), the sampling object passed (repr and normalised attributes), the original package
sampling object read before and after, the package's requested sampling, the completions (text, token ids, finish and stop reason), the INTENDED piece hashes computed from the audio before the
call and whether each captured request hash equals its intended piece hash, plus the installed package file hashes. tools/capture_replay.py checks it on the CPU.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def load_pieces(jobs_path, windows):
    """-> (pieces, {call: sha256 of the audio FILE})."""
    from meralion_worker import audio
    paths = {job["job_name"]: job["input"] for job in json.loads(Path(jobs_path).read_text(encoding="utf-8"))["jobs"]}
    cache, pieces = {}, []
    for window in windows:
        if window["call"] not in cache:
            cache[window["call"]] = audio.load_recording(window["call"], paths[window["call"]])
        pieces.append(cache[window["call"]].get(window["start"], window["end"]))
    return pieces, {call: recording.sha256 for call, recording in cache.items()}


def piece_sha256(piece):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(piece, dtype=np.float32).tobytes()).hexdigest()


def sampling_view(sampling):
    from meralion_worker.decoders.vllm_decoder import normalised_sampling
    return {"repr": repr(sampling), "attrs": normalised_sampling(sampling)}


def package_identity():
    """The installed package files the adapter's equivalence rests on, hashed (tools/capture_replay.py compares them with the verified wheel)."""
    import inspect
    from importlib import metadata
    from meralion_3_asr import prompts
    from meralion_3_asr.backends import vllm_backend
    return {"version": metadata.version("meralion-3-asr"),
            "files_sha256": {"backends/vllm_backend.py": hashlib.sha256(Path(inspect.getsourcefile(vllm_backend)).read_bytes()).hexdigest(),
                             "prompts.py": hashlib.sha256(Path(prompts.__file__).read_bytes()).hexdigest()}}


class Capture:
    """Wraps ONE llm.generate for the duration of pass 0: calls the original, records the request structure and the completions, returns the original outputs unchanged."""

    def __init__(self, llm):
        self.llm, self.records = llm, []

    def install(self):
        original = self.llm.generate

        def generate(requests, sampling_params, *args, **kwargs):
            outputs = original(requests, sampling_params, *args, **kwargs)
            rows = []
            for request in requests:
                piece, rate = request["multi_modal_data"]["audio"]
                rows.append({"prompt": request["prompt"], "audio_rate": rate, "audio_dtype": str(piece.dtype), "audio_shape": list(piece.shape), "audio_sha256": piece_sha256(piece)})
            completions = []
            for output in outputs:
                first = output.outputs[0]
                completions.append({"text": first.text, "token_ids": [int(t) for t in first.token_ids], "finish_reason": getattr(first, "finish_reason", None),
                                    "stop_reason": getattr(first, "stop_reason", None)})
            self.records.append({"requests": rows, "sampling": sampling_view(sampling_params), "completions": completions})
            return outputs

        self.llm.generate = generate

    def remove(self):
        del self.llm.generate                                                                 # the instance attribute goes; the class method is back


def decode_pass(decoder, windows, pieces, mode):
    from meralion_worker.decoders.base import DecodeRequest
    requests = [DecodeRequest(w["call"], w["start"], w["end"], p) for w, p in zip(windows, pieces)]
    results = decoder.decode_many(requests) if mode == "batch" else [decoder.decode(r.piece, recording_id=r.recording_id, start_sample=r.start_sample, end_sample=r.end_sample) for r in requests]
    return [{"text": r.text, "text_sha256": hashlib.sha256(r.text.encode("utf-8")).hexdigest(), "stop_reason": r.stop_reason, "tokens": r.tokens, "finish_reason": r.native.get("finish_reason"),
             "token_ids_sha256": r.native.get("token_ids_sha256")} for r in results]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--settings-id", required=True)
    parser.add_argument("--model-dir", default=os.environ.get("MERALION_MODEL_DIR"))
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--passes", default="batch,batch,single")
    parser.add_argument("--package-parity", action="store_true")
    parser.add_argument("--capture", type=Path, help="write the capture of the first adapter batch (follow-up E); pass 0 must be a batch pass")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0":
        print("vllm_windows: refuses to run in the diagnostic single-process mode; this tool measures the NORMAL multiprocess execution", file=sys.stderr)
        return 2
    if args.capture and args.passes.split(",")[0] != "batch":
        print("vllm_windows: --capture records the first adapter batch, so the first pass must be 'batch'", file=sys.stderr)
        return 2
    windows = json.loads(args.windows.read_text(encoding="utf-8"))["windows"]
    pieces, file_hashes = load_pieces(args.jobs, windows)
    from meralion_worker.decoders.vllm_decoder import VllmDecoder
    launcher = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "unset (normal)")
    decoder = VllmDecoder(args.model_dir, settings_id=args.settings_id)
    modes = args.passes.split(",")
    capture, package_original_before = None, None
    if args.capture:
        package_original_before = sampling_view(decoder.backend.sampling_params)
        capture = Capture(decoder.backend.llm)
    passes = []
    for index, mode in enumerate(modes):
        if capture is not None and index == 0:
            capture.install()
        try:
            results = decode_pass(decoder, windows, pieces, mode)
        finally:
            if capture is not None and index == 0:
                capture.remove()
        passes.append({"mode": mode, "results": results})
    differences = []
    for index, window in enumerate(windows):
        for other in passes[1:]:
            a, b = passes[0]["results"][index], other["results"][index]
            changed = [key for key in ("text_sha256", "stop_reason", "tokens", "token_ids_sha256") if a[key] != b[key]]
            if changed:
                differences.append({"window": window["label"], "first_pass": passes[0]["mode"], "other_pass": other["mode"], "differs_in": changed})
    report = {"tool": "vllm_windows", "mode": {"VLLM_ENABLE_V1_MULTIPROCESSING": launcher}, "execution": dict(decoder.execution), "settings_id": decoder.settings_id, "identity": decoder.identity(),
              "sampling": decoder.describe()["sampling"], "windows": windows, "passes": passes, "differences_between_passes": differences, "worker_git_commit": os.environ.get("MERALION_WORKER_GIT_COMMIT")}
    checks = {}
    first = passes[0]["results"]
    normalised = decoder.observed["normalised_sampling"]
    checks["d6_greedy_normalised_no_stop"] = normalised.get("top_k") == 0 and normalised.get("temperature") == 0 and not decoder.stops_configured
    checks["d7_finish_reasons_are_stop_or_length"] = all(r["finish_reason"] in ("stop", "length") for p in passes for r in p["results"])
    checks["token_ids_available"] = all(r["token_ids_sha256"] for p in passes for r in p["results"])
    if args.package_parity:
        backend = decoder.backend
        original = backend.sampling_params
        backend.sampling_params = decoder.sampling_params                               # HARNESS ONLY: the package's own code path under the adapter's effective parameters
        try:
            package_texts = backend.transcribe_chunks(pieces)
        finally:
            backend.sampling_params = original
        report["package_texts_sha256"] = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in package_texts]
        checks["d1_adapter_text_equals_package_text_with_the_same_parameters"] = [r["text"] for r in first] == package_texts
    if capture is not None:
        intended = [{"label": w["label"], "call": w["call"], "start": w["start"], "end": w["end"], "piece_sha256": piece_sha256(p)} for w, p in zip(windows, pieces)]
        record = capture.records[0] if capture.records else {"requests": [], "sampling": None, "completions": []}
        correspondence = len(record["requests"]) == len(intended) and [r["audio_sha256"] for r in record["requests"]] == [i["piece_sha256"] for i in intended]
        capture_document = {"format": 1, "tool": "vllm_windows", "settings_id": decoder.settings_id, "execution": dict(decoder.execution),
                            "engine_effective": decoder.describe()["engine_effective"], "constructor_overrides": dict(decoder.constructor_overrides),            # REV 3.0 B2.4: recorded, not part of the replay checks
                            "worker_git_commit": os.environ.get("MERALION_WORKER_GIT_COMMIT"), "package": package_identity(), "package_requested": dict(decoder.package_requested_sampling),
                            "package_original": package_original_before, "package_original_after": sampling_view(decoder.backend.sampling_params), "prompt": decoder.backend._prompt,
                            "intended": intended, "audio_file_sha256": file_hashes, "generate_calls_captured": len(capture.records), "call": record, "live_request_correspondence": correspondence}
        args.capture.write_text(json.dumps(capture_document, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        checks["capture_live_request_correspondence"] = correspondence and len(capture.records) == 1
    report["checks"] = checks
    report["passed"] = all(checks.values())
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    runaways = [w["label"] for w, r in zip(windows, first) if r["stop_reason"] == "max_new_tokens"]
    print(json.dumps({"settings_id": decoder.settings_id, "checks": checks, "passed": report["passed"], "max_new_tokens_windows": runaways, "pass_differences": len(differences)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
