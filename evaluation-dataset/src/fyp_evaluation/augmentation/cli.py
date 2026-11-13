"""Command line for environment_v2: init, assign, preview, freeze, build, verify (all offline)."""

import argparse
import sys
from pathlib import Path

from . import AUDIO_ASSETS, AugmentationError
from . import assign as assign_module
from . import build as build_module

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "advisory_execution_multilingual_v1"


def _fmt(value, spec):
    return "-" if value is None else format(value, spec)


def print_preview(rows):
    header = "%-24s %-11s %-9s %-7s %8s %8s %6s %8s %8s %8s" % ("clip", "language", "asset", "target", "full dB", "band dB", "AI", "asset g", "noise", "client g")
    print(header)
    for r in rows:
        print("%-24s %-11s %-9s %-7s %8s %8s %6s %8s %8s %8s" % (
            r["file"].replace(".wav", ""), r["language"], r["asset"] or "-", _fmt(r["snr_target_db"], ".1f") + ("*" if r.get("snr_clamped") else ""),
            _fmt(r["snr_fullband_db"], ".1f"), _fmt(r["snr_speechband_db"], ".1f"), _fmt(r["ai_index"], ".2f"), _fmt(r["asset_gain_db"], "+.1f"),
            _fmt(r["noise_dbfs"], ".1f"), _fmt(r["client_gain_db"], ".2f")))
        if r["client_level_change_db"] is not None:
            print("    channel: client active level %+.1f dB vs source" % r["client_level_change_db"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="describe the noise assets (assets.json) and create a DRAFT config.json")
    p = sub.add_parser("assign", help="write or check assignments.json")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    p = sub.add_parser("preview", help="short listening clips from DEVELOPMENT calls")
    p.add_argument("--only", nargs="+", metavar="COND", help="e.g. N2 D1")
    p.add_argument("--calls", nargs="+", metavar="COND=CALL_ID", help="use this DEVELOPMENT call for a condition")
    p.add_argument("--cases", nargs="+", metavar="COND:CALL_ID", help="explicit condition:call pairs (a condition may repeat on several calls)")
    p.add_argument("--tag", help="write into previews/<tag>/ so earlier rounds are kept")
    p = sub.add_parser("calibrate", help="run the real processing for every assigned call of a split; writes NO audio")
    p.add_argument("--split", choices=("DEVELOPMENT", "HELDOUT"), default="DEVELOPMENT")
    sub.add_parser("freeze", help="DRAFT -> FROZEN (pins the existing assignments.json hash)")
    p = sub.add_parser("build", help="write the non-clean variants (needs FROZEN)")
    p.add_argument("--split", choices=("DEVELOPMENT", "HELDOUT"))
    p.add_argument("--calls", nargs="+")
    p.add_argument("--rebuild", action="store_true")
    p = sub.add_parser("share", help="copy one split variants into its derived dataset folder (default: the _development copy)")
    p.add_argument("--split", choices=("DEVELOPMENT",), default="DEVELOPMENT")
    p.add_argument("--target", type=Path, help="derived dataset folder (default: <dataset-root>_development)")
    p = sub.add_parser("verify", help="re-check every built variant")
    p.add_argument("--calls", nargs="+")
    args = parser.parse_args(argv)
    root = args.dataset_root
    folder = build_module.variant_dir(root)
    try:
        if args.command == "init":
            assets_path = build_module.evaluation_root(root) / AUDIO_ASSETS / "assets.json"
            assets = build_module.scan_assets(root)
            build_module.write_json(assets_path, assets)
            print("assets.json:", len(assets["assets"]), "assets")
            if (folder / "config.json").is_file():
                print("config.json exists; left unchanged")
            else:
                build_module.write_json(folder / "config.json", build_module.default_config(assets))
                print("config.json written (DRAFT)")
        elif args.command == "assign":
            path = folder / "assignments.json"
            expected = assign_module.dump(assign_module.build_assignments(root))
            if args.write:
                if path.exists():
                    raise AugmentationError("assignments.json exists; it is written once (delete it deliberately to redraw)")
                folder.mkdir(parents=True, exist_ok=True)
                path.write_bytes(expected)
                print("assignments.json written")
            elif path.read_bytes() != expected:
                print("assignments.json differs from the deterministic recomputation", file=sys.stderr)
                return 1
            else:
                print("assignments verified: byte-identical to the deterministic recomputation")
        elif args.command == "preview":
            calls = dict(item.split("=", 1) for item in args.calls) if args.calls else None
            cases = [tuple(item.split(":", 1)) for item in args.cases] if args.cases else None
            print_preview(build_module.preview(root, only=args.only, calls=calls, cases=cases, tag=args.tag, progress=lambda m: None))
            print("clips in", folder / "previews" / (args.tag or ""))
        elif args.command == "calibrate":
            rows = build_module.calibrate(root, split=args.split, progress=lambda m: None)
            print("%-4s %-9s %-11s %5s %6s %6s %6s %6s %12s %9s %9s %9s" % ("cond", "severity", "language", "calls", "failed", "AI min",
                  "AI max", "AI avg", "SNR range", "clamped", "client g", "asset g"))
            for g in build_module.summarise_calibration(rows):
                snr_range = "-" if g["snr_min"] is None else "%.1f..%.1f" % (g["snr_min"], g["snr_max"])
                print("%-4s %-9s %-11s %5d %6d %6s %6s %6s %12s %9d %9s %9s" % (g["condition"], g["severity"] or "-", g["language"], g["calls"], g["failed"],
                      _fmt(g["ai_min"], ".2f"), _fmt(g["ai_max"], ".2f"), _fmt(g["ai_mean"], ".2f"), snr_range, g["clamped"],
                      _fmt(g["client_gain_min_db"], ".1f"), _fmt(g["asset_gain_max_db"], "+.1f")))
            print("report:", folder / "calibration_report.json")
        elif args.command == "freeze":
            build_module.freeze(root)
            print("config.json is FROZEN; assignments.json untouched")
        elif args.command == "build":
            summary = build_module.build(root, split=args.split, calls=args.calls, rebuild=args.rebuild)
            print("built variants on record:", len(summary["built"]))
        elif args.command == "share":
            target = args.target or root.with_name(root.name + "_development")
            summary = build_module.share(root, target, args.split, progress=lambda m: None)
            print("copied", summary["variants_copied"], "variants for", summary["calls"], args.split, "calls into", target)
            print(summary["clean_calls"], "clean calls need no file: they use their original recording")
        elif args.command == "verify":
            problems = build_module.verify(root, calls=args.calls, progress=lambda m: None)
            for problem in problems:
                print("problem:", problem, file=sys.stderr)
            if problems:
                return 1
            print("verified: every built variant matches the frozen config and assignments")
    except AugmentationError as error:
        print("environment_v2 error:", error, file=sys.stderr)
        return 2
    return 0
