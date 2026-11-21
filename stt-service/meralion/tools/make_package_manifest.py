"""Complete the MERaLiON package manifest from the installed package and a downloaded artifact. The provenance verdict is COMPUTED, not typed in.

    python tools/make_package_manifest.py --base configs/third_party/meralion-3-asr-0.1.0.json --package-dir <installed meralion_3_asr folder> --artifact <meralion_3_asr-0.1.0-*.whl> --out <manifest.json>
    python tools/make_package_manifest.py --base ... --package-dir <folder> --not-on-pypi --out <manifest.json>

Verdicts:
  identical_to_pypi   every file of the installed package equals the same member of the downloaded wheel, and the wheel has no other member of the package
  differs_from_pypi   the artifact was downloaded but the files differ (the differing names are printed)
  not_on_pypi         --not-on-pypi: no artifact exists to compare with; the installed files are recorded as the only reference (needs a human decision to use)
The manifest is written with status "complete" in all three cases (the files are real); the verdict says how far they can be trusted.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PACKAGE_NAME = "meralion_3_asr"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def installed_files(directory):
    return {p.relative_to(directory).as_posix(): sha256_bytes(p.read_bytes()) for p in sorted(Path(directory).rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def wheel_files(artifact):
    out = {}
    with zipfile.ZipFile(artifact) as archive:
        for name in archive.namelist():
            if name.startswith(f"{PACKAGE_NAME}/") and not name.endswith("/") and "__pycache__" not in name:
                out[name[len(PACKAGE_NAME) + 1:]] = sha256_bytes(archive.read(name))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--not-on-pypi", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if bool(args.artifact) == bool(args.not_on_pypi):
        parser.error("give exactly one of --artifact or --not-on-pypi")
    base = json.loads(args.base.read_text(encoding="utf-8"))
    files = installed_files(args.package_dir)
    if not files:
        print("REFUSED: no files in the package directory", file=sys.stderr)
        return 1
    if args.not_on_pypi:
        verdict, artifact, differing = "not_on_pypi", None, []
    else:
        reference = wheel_files(args.artifact)
        differing = sorted({n for n in files if reference.get(n) != files[n]} | {n for n in reference if n not in files})
        verdict = "identical_to_pypi" if not differing else "differs_from_pypi"
        artifact = {"filename": args.artifact.name, "sha256": sha256_bytes(args.artifact.read_bytes())}
    manifest = {k: v for k, v in base.items() if k not in ("status_note", "hints")}
    manifest.update(status="complete", provenance_verdict=verdict, artifact=artifact, files=files)
    args.out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"verdict {verdict}; {len(files)} files" + (f"; differing: {differing}" if differing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
