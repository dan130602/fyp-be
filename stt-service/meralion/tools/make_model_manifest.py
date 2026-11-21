"""Complete a model manifest from real files. Never invents values: it hashes what exists and cross-checks what is already recorded.

    python tools/make_model_manifest.py --base configs/models/meralion-3-3b-asr-base-9f226d7.json --dir <checkpoint folder> --out <manifest.json>
    python tools/make_model_manifest.py --base ... --listing <sha256sum output> --sizes <'<bytes> <name>' lines> --out <manifest.json>

Rules:
  * The files are hashed from --dir (every file except a `.cache` folder), or read from a sha256sum-format listing plus a size listing.
  * Every file already recorded in the base manifest must match its recorded sha256 and size.
  * Every file named under `missing` must be present, and its recorded first-8 / last-3 hash hints must agree (hints are only a sanity check).
  * The result is status "complete" only if all of the above hold; otherwise nothing is written and the exit code is 1.
  * The manifest always needs `adapter_config.json` to be absent.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

CHUNK = 8 * 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def from_directory(directory):
    files = {}
    for path in sorted(Path(directory).rglob("*")):
        relative = path.relative_to(directory)
        if not path.is_file() or ".cache" in relative.parts:
            continue
        files[relative.as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return files


def from_listing(listing, sizes):
    hashes = {}
    for line in Path(listing).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, _, name = line.partition(" ")
            hashes[name.strip().lstrip("*")] = digest
    sized = {}
    for line in Path(sizes).read_text(encoding="utf-8").splitlines():
        if line.strip():
            size, _, name = line.strip().partition(" ")
            sized[name.strip()] = int(size)
    if set(hashes) != set(sized):
        raise SystemExit(f"listing and sizes name different files: {sorted(set(hashes) ^ set(sized))}")
    return {name.removeprefix("./"): {"sha256": hashes[name], "bytes": sized[name]} for name in hashes}


def build(base, files, source):
    problems = []
    if "adapter_config.json" in files:
        problems.append("adapter_config.json is present: complete checkpoints only")
    for name, recorded in base.get("files", {}).items():
        found = files.get(name)
        if found is None:
            problems.append(f"recorded file not found: {name}")
        elif found["sha256"] != recorded["sha256"] or found["bytes"] != recorded["bytes"]:
            problems.append(f"recorded value differs for {name}")
    for entry in base.get("missing", []):
        found = files.get(entry["name"])
        if found is None:
            problems.append(f"file named under 'missing' is not present: {entry['name']}")
        elif not (found["sha256"].startswith(entry["hint_first8"]) and found["sha256"].endswith(entry["hint_last3"])):
            problems.append(f"hash hints disagree for {entry['name']}")
    if not any(name.endswith(".safetensors") for name in files):
        problems.append("no safetensors files")
    if problems:
        return None, problems
    manifest = {k: v for k, v in base.items() if k not in ("missing", "hint_source", "status_note")}
    manifest["status"] = "complete"
    manifest["files_listing_complete"] = True
    manifest["files"] = {name: {**record, "source": source} for name, record in sorted(files.items())}
    return manifest, []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dir", type=Path)
    source.add_argument("--listing", type=Path)
    parser.add_argument("--sizes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    base = json.loads(args.base.read_text(encoding="utf-8"))
    if args.dir:
        files, origin = from_directory(args.dir), f"sha256 computed from the directory {args.dir.name}"
    else:
        if not args.sizes:
            parser.error("--listing needs --sizes")
        files, origin = from_listing(args.listing, args.sizes), f"sha256sum listing {args.listing.name}"
    manifest, problems = build(base, files, origin)
    if problems:
        for problem in problems:
            print("REFUSED:", problem, file=sys.stderr)
        return 1
    args.out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"complete manifest with {len(manifest['files'])} files -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
