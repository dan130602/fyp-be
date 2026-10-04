"""Save the per-turn SSML used by render_polly.py; makes no AWS requests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from render_polly import CALLS, load_turns, make_ssml


def export_call(call_id: str, render_id: str, *, overwrite: bool = False) -> int:
    turns = load_turns(call_id)
    output_dir = CALLS / call_id / "renders" / render_id
    segment_dir = output_dir / "segments"
    manifest_path = output_dir / "ssml_manifest.json"
    files = [(segment_dir / f"{turn['turn_id']}.ssml", make_ssml(turn["text"]))
             for turn in turns]
    manifest = {
        "schema_version": "0.1.0",
        "call_id": call_id,
        "render_id": render_id,
        "text_type": "ssml",
        "turns": [{"turn_id": turn["turn_id"], "speaker": turn["speaker"],
                   "ssml_file": f"segments/{turn['turn_id']}.ssml"} for turn in turns],
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    files.append((manifest_path, manifest_text))
    conflicts = [path for path, content in files if path.exists()
                 and path.read_text(encoding="utf-8") != content]
    if conflicts and not overwrite:
        raise FileExistsError(f"SSML differs from the canonical dialogue: {conflicts[0]}; pass --overwrite to replace it")
    segment_dir.mkdir(parents=True, exist_ok=True)
    for path, content in files:
        if not path.exists() or (overwrite and path in conflicts):
            path.write_text(content, encoding="utf-8")
    return len(turns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all", action="store_true")
    choice.add_argument("--call")
    parser.add_argument("--render-id", default="clean_voice_A")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.call and not re.fullmatch(r"CALL_[0-9]{3,}", args.call):
        parser.error("--call must be a canonical ID such as CALL_001")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.render_id):
        parser.error("Invalid --render-id")
    call_ids = [p.name for p in sorted(CALLS.glob("CALL_*")) if p.is_dir()] if args.all else [args.call]
    try:
        for call_id in call_ids:
            count = export_call(call_id, args.render_id, overwrite=args.overwrite)
            print(f"{call_id}: saved {count} SSML files and ssml_manifest.json")
    except (OSError, ValueError) as error:
        parser.exit(1, f"SSML export failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
