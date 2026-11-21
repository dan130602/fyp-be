"""HISTORICAL: stock settings only; NOT used to validate profiles (REV 3.0 A1). Use device_smoke.py, which requires --settings-id and makes no adapter-versus-package equality check.

J2 part (i): a live check of the vLLM adapter on the first FG windows of one recording (run on Origami inside the vLLM environment; never part of the image).

    python vllm_smoke.py --model-dir <model> --audio <wav> --call CALL_064 --windows 4 --out smoke.json

Checks, each recorded and the exit code non-zero unless every one holds:
  d1   the adapter's decode_many text equals the package's own backend.transcribe_chunks text for the same windows (live D1 parity)
  seq  sequential decode() text equals the batched decode_many() text
  d6   the constructed sampling object has top_k 0, temperature 0 and no stop configured (live D6), next to the requested package dict
  d7   every native finish_reason is "stop" or "length", and the mapped value agrees with the D7 rules (live D7)
Also recorded: describe()["settings"], repr(backend.sampling_params), per-window native values, tokens, peak GPU memory.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--call", required=True)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from meralion_worker import audio, strategy_fg
    from meralion_worker.decoders.base import DecodeRequest
    from meralion_worker.decoders.vllm_decoder import VllmDecoder, clean_text, map_stop_reason

    recording = audio.load_recording(args.call, args.audio)
    bounds = strategy_fg.windows_for(recording.num_samples)[: args.windows]
    requests = [DecodeRequest(args.call, a, b, recording.get(a, b)) for a, b in bounds]
    started = time.monotonic()
    decoder = VllmDecoder(args.model_dir)
    load_seconds = round(time.monotonic() - started, 1)
    backend = decoder.backend

    batched = decoder.decode_many(requests)
    sequential = [decoder.decode(r.piece, recording_id=r.recording_id, start_sample=r.start_sample, end_sample=r.end_sample) for r in requests]
    package_texts = backend.transcribe_chunks([r.piece for r in requests])               # the package's own code path, same backend object, same windows
    describe = decoder.describe()
    settings = describe["settings"]
    normalised = decoder.observed["normalised_sampling"]

    rows = []
    for (a, b), got, seq in zip(bounds, batched, sequential):
        rows.append({"window": [a, b], "text": got.text, "tokens": got.tokens, "mapped_stop_reason": got.stop_reason, "native": got.native,
                     "mapped_matches_d7": got.stop_reason == map_stop_reason(got.native["finish_reason"], got.native["stop_reason"], decoder.stops_configured),
                     "sequential_equal": seq.text == got.text})
    peak = None
    try:
        import torch
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None
    except Exception:
        pass
    checks = {
        "d1_adapter_text_equals_package_text": [r.text for r in batched] == [clean_text(t) for t in package_texts] == package_texts,
        "seq_sequential_equals_batched": all(row["sequential_equal"] for row in rows),
        "d6_top_k_0_temperature_0_no_stop": normalised.get("top_k") == 0 and normalised.get("temperature") == 0 and not decoder.stops_configured,
        "d7_finish_reasons_are_stop_or_length": all(row["native"]["finish_reason"] in ("stop", "length") for row in rows) and all(row["mapped_matches_d7"] for row in rows),
    }
    report = {"call": args.call, "windows": rows, "model_load_s": load_seconds, "checks": checks, "passed": all(checks.values()), "describe_settings": settings,
              "requested_sampling": decoder.observed["requested_sampling"], "normalised_sampling": normalised, "sampling_params_repr": repr(backend.sampling_params),
              "package_texts": package_texts, "peak_torch_allocated_gib": peak, "versions": describe["versions"], "device": describe["device"]}
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "passed": report["passed"], "mapped": [r["mapped_stop_reason"] for r in rows], "tokens": [r["tokens"] for r in rows]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
