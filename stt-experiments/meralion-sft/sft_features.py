"""Per-clip processor-output cache for the SFT trainer (CPU; no GPU, no model weights).

    python -u sft_features.py build  --data <sft-data-v2> --batch <audio batch dir> --model <model dir> --out <cache dir>
    python -u sft_features.py verify --data <sft-data-v2> --batch <audio batch dir> --model <model dir> --out <cache dir>

Why it is safe. The trainer builds every example by calling the processor on the clip's 16 kHz slice (chat prompt with the speech placeholders expanded, log-mel input
features, feature mask). That output is a pure function of (audio slice, prompt, processor). The cache stores exactly that output, float32 as the processor returns it
(the model casts to bf16 later, as before), and every entry is bound to
  * the SHA-256 of the exact audio slice the trainer is about to use (a changed slice, boundary or audio file is detected and refused, never silently reused),
  * the SHA-256 of the data package manifest and of the chat prompt,
  * the SHA-256 of its shard file (a damaged or edited shard is refused).
`verify` recomputes EVERY clip with the live processor and compares with torch.equal (shape, dtype and every value); the cache is only to be used when it reports 0 mismatches.
Sampling, targets, loss, seeds and the model are untouched: only where the processor output comes from changes.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

VERSION = 1
SHARD_CLIPS = 256
KEYS = ("input_ids", "input_features", "feature_attention_mask")


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def slice_sha256(audio):
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def processor_signature(processor):
    """What of the processor configuration changes its output beyond the prompt and the audio: the number of speech placeholders per 30 s chunk (300 for the pinned 5x model, 500 for the 3x adapter)."""
    return f"fixed_speech_embeds_length={int(getattr(processor, 'fixed_speech_embeds_length', 300))}"


def compute(processor, prompt, audio, load_audio):
    """The trainer's own processor call (sft_core.build_example), returned as detached CPU tensors."""
    out = processor(text=[prompt], audios=[load_audio(audio)], return_tensors="pt", padding=True)
    return {k: out[k].detach().clone().cpu() for k in KEYS}


class FeatureCache:
    def __init__(self, folder, index):
        self.folder, self.index = Path(folder), index
        self.entries = index["clips"]
        self.shards = {}

    @classmethod
    def open(cls, folder, manifest_sha256, prompt, signature=None):
        folder = Path(folder)
        index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        if index.get("version") != VERSION:
            raise ValueError("feature cache: unsupported version")
        if index["manifest_sha256"] != manifest_sha256:
            raise ValueError("feature cache: built for a different data package manifest")
        if index["prompt_sha256"] != text_sha256(prompt):
            raise ValueError("feature cache: built with a different chat prompt")
        if signature is not None and index.get("processor_signature", "fixed_speech_embeds_length=300") != signature:      # caches built before this key existed are 5x caches
            raise ValueError(f"feature cache: built with a different processor configuration ({index.get('processor_signature', 'fixed_speech_embeds_length=300')} vs {signature})")
        if not index.get("verified"):
            raise ValueError("feature cache: not marked verified (run sft_features.py verify first)")
        return cls(folder, index)

    def shard(self, name):
        if name not in self.shards:
            path = self.folder / name
            if sha256_file(path) != self.index["shards"][name]:
                raise ValueError(f"feature cache: shard {name} differs from the index (damaged or edited)")
            self.shards[name] = torch.load(path, map_location="cpu", weights_only=True)
        return self.shards[name]

    def get(self, clip, audio):
        entry = self.entries.get(clip["clip_id"])
        if entry is None:
            raise ValueError(f"feature cache: no entry for {clip['clip_id']}")
        if slice_sha256(audio) != entry["audio_sha256"]:
            raise ValueError(f"feature cache: the audio slice of {clip['clip_id']} differs from the one the cache was built from")
        item = self.shard(entry["shard"])[clip["clip_id"]]
        return {k: item[k].clone() for k in KEYS}


def build_from(clips, store, processor, prompt, load_audio, out, manifest_sha256, log=print):
    """Compute and store the processor output of every clip (train then validation order), then write index.json (unverified)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    clips = list(clips)
    entries, shards, shard, start = {}, {}, {}, time.time()
    for i, clip in enumerate(clips):
        audio = store.clip(clip)
        shard[clip["clip_id"]] = compute(processor, prompt, audio, load_audio)
        entries[clip["clip_id"]] = {"shard": f"shard-{len(shards):04d}.pt", "audio_sha256": slice_sha256(audio)}
        if len(shard) == SHARD_CLIPS or i == len(clips) - 1:
            name = f"shard-{len(shards):04d}.pt"
            torch.save(shard, out / name)
            shards[name] = sha256_file(out / name)
            shard = {}
        if (i + 1) % 200 == 0 or i == len(clips) - 1:
            log(f"built {i + 1}/{len(clips)} clips, {time.time() - start:.0f}s")
    index = {"version": VERSION, "manifest_sha256": manifest_sha256, "prompt_sha256": text_sha256(prompt), "processor_signature": processor_signature(processor), "clips": entries, "shards": shards, "verified": False}
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def verify_from(clips, store, processor, prompt, load_audio, out, manifest_sha256, log=print):
    """Recompute every clip live and compare bit-for-bit; mark the index verified only when there are no mismatches. Returns the report."""
    out = Path(out)
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    index["verified"] = True                                            # lets FeatureCache.open read it; reset below unless everything matches
    cache = FeatureCache(out, index)
    if index["manifest_sha256"] != manifest_sha256 or index["prompt_sha256"] != text_sha256(prompt):
        raise ValueError("feature cache: manifest or prompt differs from the ones it was built with")
    if index.get("processor_signature", "fixed_speech_embeds_length=300") != processor_signature(processor):
        raise ValueError("feature cache: the processor configuration differs from the one it was built with")
    clips = list(clips)
    mismatches, start = [], time.time()
    for i, clip in enumerate(clips):
        audio = store.clip(clip)
        live, cached = compute(processor, prompt, audio, load_audio), cache.get(clip, audio)
        for key in KEYS:
            same = live[key].dtype == cached[key].dtype and live[key].shape == cached[key].shape and torch.equal(live[key], cached[key])
            if not same:
                mismatches.append({"clip_id": clip["clip_id"], "key": key})
        if (i + 1) % 200 == 0 or i == len(clips) - 1:
            log(f"verified {i + 1}/{len(clips)} clips, {time.time() - start:.0f}s, mismatches so far {len(mismatches)}")
    report = {"clips": len(clips), "keys_compared": list(KEYS), "mismatches": len(mismatches), "first_mismatches": mismatches[:10], "bit_identical": not mismatches}
    index["verified"] = not mismatches
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    (out / "verify.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("build", "verify"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    import sft_train as T
    manifest = (args.data / "manifest.json").read_bytes()
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    for name, digest in json.loads(manifest)["files"].items():
        if hashlib.sha256((args.data / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{name} differs from the frozen manifest")
    clips = [json.loads(l) for name in ("clips_train.jsonl", "clips_validation.jsonl") for l in (args.data / name).read_text(encoding="utf-8").splitlines()]
    processor, prompt, load_audio = T.load_processor(args.model)
    store = T.WaveStore(args.batch, {c["call_id"] for c in clips}, load_audio)
    if args.command == "build":
        build_from(clips, store, processor, prompt, load_audio, args.out, manifest_sha)
        print(f"built {len(clips)} clips into {args.out} (not yet verified: run verify)")
        return 0
    report = verify_from(clips, store, processor, prompt, load_audio, args.out, manifest_sha)
    print(json.dumps(report))
    return 0 if report["bit_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
