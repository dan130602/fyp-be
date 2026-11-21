"""Turn worker output documents into prediction files the unchanged scorer path can import (Stage 3; never part of the image; reads no reference text).

    python tools/to_predictions.py --docs <transcripts folder> --variant raw|final --out <new folder>

One CALL_x.json per document in the format of evaluation-engine/transcribe_eval/predictions.py: call_id, settings (identical across the calls of a run except audio_sha256), transcript, audio_seconds.
`raw` writes meralion.transcript_raw and `final` writes meralion.transcript_final (what results.transcripts[0].transcript holds). No chunks are written: the worker's windows overlap, so they are not
a contiguous chunk record. Nothing is edited, formatted or invented here.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

VARIANTS = ("raw", "final")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def settings_of(document, variant):
    block = document["meralion"]
    decoder = block["decoder"]
    dtype = decoder.get("dtype") or (decoder.get("settings") or {}).get("dtype")
    return {"revision": block["model"]["revision"], "adapter": None, "training": False, "strategy": f"worker-v0-{decoder['backend']}-{block['config']['config_id']}", "maximum_seconds": 30,
            "dtype": dtype if dtype is not None else "not recorded", "implementation_sha256": digest(block["worker"]["vendored"]), "backend": decoder["backend"], "settings_id": decoder["settings_id"],
            "config_fingerprint": block["config_fingerprint"], "transcript_variant": variant, "audio_sha256": block["audio"]["sha256"]}


def convert(docs_dir, variant, out_dir):
    if variant not in VARIANTS:
        raise SystemExit(f"--variant must be one of {VARIANTS}")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"output folder is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path in sorted(Path(docs_dir).glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        block = document["meralion"]
        text = block["transcript_raw"] if variant == "raw" else block["transcript_final"]
        row = {"call_id": document["jobName"], "settings": settings_of(document, variant), "transcript": text, "audio_seconds": block["audio"]["seconds"]}
        (out_dir / f"{document['jobName']}.json").write_text(json.dumps(row, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(document["jobName"])
    if not written:
        raise SystemExit(f"no output documents in {docs_dir}")
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    written = convert(args.docs, args.variant, args.out)
    print(json.dumps({"written": len(written), "variant": args.variant, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
