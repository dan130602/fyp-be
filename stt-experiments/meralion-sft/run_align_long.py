"""Forced-align the KNOWN reference text of each long turn against its audio (Origami; aligner venv; GPU or CPU). Reuses the project's Qwen3-ForcedAligner-0.6B setup (alignment-pilot-20261114).

    python -u run_align_long.py --inputs long_turn_inputs.json --batch ~/meralion-pilot/batch --model ~/meralion-pilot/models/forced-aligner --out alignments.json [--device cuda:0|cpu]

The reference text is never edited; no ASR is run. Output per turn: aligner units (text, start, end in seconds relative to the turn), a 10 ms RMS track of the turn audio, the audio sha256 check, timing.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from qwen_asr import Qwen3ForcedAligner
    items = json.loads(args.inputs.read_text(encoding="utf-8"))
    if args.limit:
        items = items[:args.limit]
    calls = {c["call_id"]: c for c in json.loads((args.batch / "calls.json").read_text(encoding="utf-8"))["calls"]}
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    started = time.monotonic()
    aligner = Qwen3ForcedAligner.from_pretrained(str(args.model), dtype=dtype, device_map=args.device)
    aligner.model.eval()
    print(f"aligner loaded in {time.monotonic() - started:.1f}s on {args.device}", flush=True)
    waves, out = {}, {}
    for n, item in enumerate(items, 1):
        key = f"{item['call_id']}:{item['turn_id']}"
        t0 = time.monotonic()
        try:
            if item["call_id"] not in waves:
                row = calls[item["call_id"]]
                path = args.batch / row["audio"]
                with open(path, "rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != row["audio_sha256"]:
                        raise ValueError("audio hash differs from the batch manifest")
                samples, rate = sf.read(path, dtype="float32", always_2d=True)
                wave = samples.mean(axis=1)
                if rate != 16000:
                    g = math.gcd(rate, 16000)
                    wave = resample_poly(wave, 16000 // g, rate // g)
                waves = {item["call_id"]: np.ascontiguousarray(wave, dtype=np.float32)}       # one call in memory at a time
            wave = waves[item["call_id"]]
            piece = wave[item["start_sample"]:min(item["end_sample"], len(wave))]
            if len(piece) < 16000:
                raise ValueError("turn audio shorter than 1 s")
            units = aligner.align(audio=(piece, 16000), text=item["text"], language="English")[0]
            frames = len(piece) // 160
            rms = np.sqrt((piece[:frames * 160].reshape(frames, 160) ** 2).mean(axis=1))
            out[key] = {"units": [{"text": u.text, "start_time": float(u.start_time), "end_time": float(u.end_time)} for u in units], "rms": [round(float(x), 6) for x in rms],
                        "seconds": len(piece) / 16000, "align_seconds": round(time.monotonic() - t0, 2), "audio_sha256_ok": True}
        except Exception as error:                                   # noqa: BLE001 - a failed turn is reported as unresolved by the segmenter
            out[key] = {"error": f"{type(error).__name__}: {error}"}
        print(f"[{n}/{len(items)}] {key} {'ok' if 'units' in out[key] else out[key]['error']} ({time.monotonic() - t0:.1f}s)", flush=True)
        args.out.write_text(json.dumps(out), encoding="utf-8")
    print(f"ALIGNMENT DONE {sum('units' in v for v in out.values())}/{len(items)} in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
