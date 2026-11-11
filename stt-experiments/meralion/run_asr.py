"""Inference-only MERaLiON boundary experiment. Never loads an adapter or trains."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from segmentation import RATE, chunk_bounds
from io_utils import write_json

REVISION = "9f226d738dee62c60a2f43b0ab7b9a2585a132c3"


def guarded_bounds(waveform, maximum_seconds=30):
    """Search the last eight seconds for a sustained pause, preserving every sample.

    Use the latest qualifying 240 ms low-energy window and cut at its centre.
    If no pause exists, retain the hard limit and explicitly flag that cut.
    This is an experimental rule, not a speech/word boundary guarantee.
    """
    frame = RATE // 50
    maximum = int(maximum_seconds * RATE)
    start = 0
    while start < len(waveform):
        end = min(start + maximum, len(waveform))
        reason = "end_of_audio"
        if end < len(waveform):
            search_start = max(start, end - 8 * RATE)
            search = waveform[search_start:end]
            complete = len(search) // frame * frame
            power = np.mean(search[:complete].reshape(-1, frame) ** 2, axis=1)
            threshold = max(0.001, min(0.01, 0.35 * float(np.median(np.sqrt(power)))))
            # Every frame must be quiet, rather than just the mean of the window.
            quiet = np.sqrt(power) <= threshold
            candidates = np.flatnonzero(np.convolve(quiet.astype(int), np.ones(12, dtype=int), "valid") == 12)
            reason = "hard_limit_no_safe_pause"
            if len(candidates):
                end = search_start + (int(candidates[-1]) + 6) * frame
                reason = "sustained_pause"
        yield start, end, reason
        start = end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    choose = parser.add_mutually_exclusive_group(required=True)
    choose.add_argument("--calls", nargs="+")
    choose.add_argument("--all", action="store_true")
    parser.add_argument("--maximum-seconds", type=int, choices=(24, 30), default=30)
    parser.add_argument("--strategy", choices=("guarded", "energy"), default="guarded")
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    import soundfile
    import torch
    from meralion_3_asr import Meralion3ASR
    from meralion_3_asr._audio_io import load_audio

    manifest = json.loads((args.batch / "calls.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != "advisory_execution_multilingual_v1_development" or len(manifest["calls"]) != 175:
        raise ValueError("Only the 175-call development manifest is supported")
    if manifest["revision"] != REVISION:
        raise ValueError("The batch does not use the pinned base revision")
    if (args.model / "adapter_config.json").exists() or not (args.model / "config.json").is_file():
        raise ValueError("--model must be the standalone base checkpoint, never an adapter")
    calls = {row["call_id"]: row for row in manifest["calls"]}
    if len(calls) != len(manifest["calls"]):
        raise ValueError("Duplicate call IDs")
    selected = list(calls) if args.all else list(dict.fromkeys(args.calls))
    if not selected or set(selected) - set(calls):
        raise ValueError("Only calls in the development audio manifest are allowed")
    if args.strategy == "energy" and args.maximum_seconds != 30:
        parser.error("The unchanged energy control requires --maximum-seconds 30")
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU allocation is required for inference")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    settings = dict(revision=REVISION, adapter=None, training=False, stock_prompt=True,
                    strategy=args.strategy, maximum_seconds=args.maximum_seconds,
                    dtype=str(dtype), implementation_sha256=code_hash)
    args.output.mkdir(parents=True, exist_ok=True)
    model = None
    began = time.monotonic()
    for index, cid in enumerate(selected, 1):
        row = calls[cid]
        path = args.batch / row["audio"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["audio_sha256"]:
            raise ValueError(f"Audio hash mismatch: {cid}")
        target = args.output / f"{cid}.json"
        effective = {**settings, "audio_sha256": digest}
        if target.exists():
            saved = json.loads(target.read_text(encoding="utf-8"))
            if saved["settings"] != effective or saved["call_id"] != cid:
                raise ValueError(f"Refusing to reuse different settings: {cid}")
        else:
            samples, rate = soundfile.read(path, dtype="float32")
            waveform = load_audio((samples, rate))
            if not len(waveform) or not np.isfinite(waveform).all():
                raise ValueError(f"Invalid waveform: {cid}")
            if model is None:
                model = Meralion3ASR.from_pretrained(str(args.model.resolve()), backend="transformers",
                                                   device="cuda", batch_size=1, dtype=dtype)
                base = model._backend.model
                if any("lora_" in name for name, _ in base.named_parameters()):
                    raise ValueError("Adapter parameters detected in the base-model experiment")
                base.requires_grad_(False)
                base.eval()
            if args.strategy == "guarded":
                bounds = list(guarded_bounds(waveform, args.maximum_seconds))
            else:
                bounds = [(a, b, "original_energy_control") for a, b in chunk_bounds(waveform, True)]
            chunks = []
            for number, (start, end, reason) in enumerate(bounds, 1):
                with torch.inference_mode():
                    text = model.transcribe(waveform[start:end])
                chunks.append(dict(start_sample=start, end_sample=end, boundary_reason=reason, text=text))
                print(f"{cid} {number}/{len(bounds)} chunks", flush=True)
            transcript = " ".join(chunk["text"] for chunk in chunks if chunk["text"])
            if not transcript.strip():
                raise ValueError(f"Empty result: {cid}")
            write_json(target, dict(call_id=cid, settings=effective, chunks=chunks,
                                    transcript=transcript, audio_seconds=len(waveform) / RATE))
        progress = int(30 * index / len(selected))
        print(f"[{'#' * progress}{'-' * (30-progress)}] {index}/{len(selected)} calls; "
              f"elapsed {(time.monotonic()-began)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
