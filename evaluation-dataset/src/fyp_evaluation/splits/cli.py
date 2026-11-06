"""Command line for the split: --write-registry (local-only evidence), --write (manifest + report), --check (offline verification)."""

import argparse
import json
import sys
from pathlib import Path

from . import REGISTRY_FILE, SPLIT_DIR, SplitError
from .build import dataset_id_of, verify, write_documents
from .features import dump
from .registry import build_registry, validate_registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--schema-dir", type=Path, default=Path(__file__).resolve().parents[3] / "schemas")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-registry", action="store_true", help="build the development-use registry (reads local-only provenance)")
    mode.add_argument("--write", action="store_true", help="freeze split.v1.json and split_report.v1.json from the committed registry")
    mode.add_argument("--check", action="store_true", help="verify the committed split files; needs no ignored authoring files")
    parser.add_argument("--overwrite", action="store_true", help="replace existing split files")
    args = parser.parse_args(argv)
    root = args.dataset_root
    folder = root / SPLIT_DIR
    try:
        if args.check:
            errors = verify(root, args.schema_dir)
            for error in errors:
                print(f"split error: {error}", file=sys.stderr)
            if errors:
                return 1
            print("split verified: assignments, quotas, locks, fingerprints and deterministic reconstruction all hold")
            return 0
        existing = [p for p in (folder / REGISTRY_FILE, folder / "split.v1.json", folder / "split_report.v1.json") if p.exists()]
        if args.write_registry:
            if (folder / REGISTRY_FILE).exists() and not args.overwrite:
                raise SplitError("registry exists; pass --overwrite to replace it (locks can only be added)")
            registry = build_registry(root, dataset_id_of(root))
            errors = validate_registry(registry)
            if errors:
                raise SplitError("; ".join(errors))
            folder.mkdir(exist_ok=True)
            (folder / REGISTRY_FILE).write_text(dump(registry), encoding="utf-8", newline="\n")
            print(f"registry written: {len(registry['locks'])} locks ({len(registry['audit_derived_locks'])} audit-derived)")
            return 0
        if any(p.name != REGISTRY_FILE for p in existing) and not args.overwrite:
            raise SplitError("split files exist; the split is frozen and is never redrawn (pass --overwrite only for a deliberate new version)")
        registry = json.loads((folder / REGISTRY_FILE).read_text(encoding="utf-8"))
        manifest, report = write_documents(root, registry, args.schema_dir)
        print(f"split written: {manifest['totals']}; held-out LONG {report['long_heldout_total']}; swaps {report['search']['swaps']}")
        return 0
    except (SplitError, OSError, KeyError) as error:
        print(f"split error: {error}", file=sys.stderr)
        return 2
