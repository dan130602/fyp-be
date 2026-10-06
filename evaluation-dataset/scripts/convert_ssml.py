"""Convert plain text or canonical call dialogue to SSML without AWS requests."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.etree import ElementTree

from export_ssml import export_call
from render_polly import DEFAULT_DATASET_ROOT, make_ssml


def convert_text(text: str) -> str:
    """Preserve the spoken text while producing a valid SSML speak document."""
    if not text.strip():
        raise ValueError("Input text must not be empty")
    ssml = make_ssml(text)
    try:
        ElementTree.fromstring(ssml)
    except ElementTree.ParseError as error:
        raise ValueError("Input contains characters that are not valid in XML") from error
    return ssml


def save_ssml(path: Path, ssml: str, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing file even if it appears after parsing.
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as output:
        output.write(ssml + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Plain text to convert; prints SSML unless --output is given")
    source.add_argument("--input", type=Path, help="UTF-8 text file; defaults to a sibling .ssml file")
    source.add_argument("--call", help="Canonical call ID, for example CALL_001")
    source.add_argument("--all", action="store_true", help="Convert all canonical calls, one SSML file per turn")
    parser.add_argument("--output", type=Path, help="Destination SSML file for --text or --input")
    parser.add_argument("--render-id", default="clean_voice_A", help="Render folder for --call or --all")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing SSML outputs")
    args = parser.parse_args(argv)

    if args.output and (args.call or args.all):
        parser.error("--output is only used with --text or --input")
    if args.call and not re.fullmatch(r"CALL_[0-9]{3,}", args.call):
        parser.error("--call must be a canonical ID such as CALL_001")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.render_id):
        parser.error("Invalid --render-id")

    try:
        if args.call or args.all:
            calls_dir = args.dataset_root.resolve() / "calls"
            calls = ([args.call] if args.call else
                     [path.name for path in sorted(calls_dir.glob("CALL_*")) if path.is_dir()])
            if not calls:
                raise ValueError("No canonical calls found")
            for call_id in calls:
                count = export_call(call_id, args.render_id, calls_dir, overwrite=args.overwrite)
                print(f"{call_id}: {count} SSML files and ssml_manifest.json under "
                      f"{calls_dir / call_id / 'renders' / args.render_id}")
        else:
            text = args.input.read_text(encoding="utf-8-sig") if args.input else args.text
            ssml = convert_text(text)
            output = args.output or (args.input.with_suffix(".ssml") if args.input else None)
            if output is None:
                print(ssml)
            else:
                if output.suffix.lower() != ".ssml":
                    raise ValueError("Output must have a .ssml extension")
                if args.input and output.resolve() == args.input.resolve():
                    raise ValueError("Output must differ from the input file")
                save_ssml(output, ssml, overwrite=args.overwrite)
                print(f"Saved {output}")
    except FileExistsError as error:
        parser.exit(1, f"SSML conversion failed: {error}; use --overwrite to replace SSML outputs\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"SSML conversion failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
