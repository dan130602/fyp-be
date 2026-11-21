"""A PORTABLE comparison of two Python environments (REV 3.2 Amendment 7). Standard library only, so it runs in either environment.

    python env_identity.py --record environment-reference.json [--freeze-file vllm-pip-freeze.txt]          (once, on the cluster, in .venv-vllm)
    python env_identity.py --check environment-reference.json --package-manifest configs/third_party/meralion-3-asr-0.1.0.json --verify-environment-exit 0 [--expect-patches none|original|patched] --out environment-check.json

Why not the hash of `pip freeze`: the saved freeze has one machine-specific line, `meralion-3-asr @ file:///common/.../meralion_3_asr-0.1.0-py3-none-any.whl#sha256=...`, whose path text changes when the
identical wheel is installed elsewhere; and identical freeze text proves nothing about the installed FILES. So this compares only: package names and pinned versions; the wheel's artifact hash (against the
reference AND against the verified package manifest); the torch version and CUDA build; the audio libraries (including the bundled libsndfile); the tracked patch states; and requires that
`verify-environment` (per-file hashes of the installed package against the manifest) exited 0 on this install. REV 4.1 B6: it also enforces the deployment TARGET: operating system Linux, machine x86_64,
Python 3.12.x and the same Python minor version as the reference (the pinned wheels are cp312). Paths, host, kernel, the Python PATCH version and the freeze text are recorded and NEVER compared.
pip, setuptools and wheel are installer tooling, not part of the worker's runtime: they are listed as ignored (reported, never gated).
"""
import argparse
import hashlib
import json
import platform
import re
import sys
from importlib import metadata
from pathlib import Path

IGNORED = ("pip", "setuptools", "wheel")
TARGET_SYSTEM, TARGET_MACHINE, TARGET_PYTHON_MINOR = "Linux", "x86_64", "3.12"            # REV 4.1 B6: the deployment target (the cp312 wheels of the pinned stack, an x86-64 Linux host)
AUDIO_PACKAGES = ("soundfile", "librosa", "soxr", "numpy", "scipy")
PACKAGE = "meralion-3-asr"


def normalise(name):
    """PEP 503 normalisation."""
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_pins():
    pins = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            pins[normalise(name)] = distribution.version
    return pins


def artifact_hash_of(package=PACKAGE):
    """The sha256 of the wheel the package was installed from (dist-info/direct_url.json archive_info.hash), None when it was not installed from a file."""
    try:
        distribution = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return None
    text = distribution.read_text("direct_url.json")
    if not text:
        return None
    archive = (json.loads(text).get("archive_info") or {})
    value = archive.get("hash") or ""
    if value.startswith("sha256="):
        return value[len("sha256="):]
    hashes = archive.get("hashes") or {}
    return hashes.get("sha256")


def torch_facts():
    try:
        import torch
        return {"version": torch.__version__, "cuda": torch.version.cuda}
    except Exception as error:                                                                  # a missing torch is a recorded fact, not a crash
        return {"version": None, "cuda": None, "error": str(error)[:120]}


def audio_facts(pins):
    facts = {name: pins.get(normalise(name)) for name in AUDIO_PACKAGES}
    try:
        import soundfile
        facts["libsndfile"] = soundfile.__libsndfile_version__
    except Exception:
        facts["libsndfile"] = None
    return facts


def patch_states():
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from meralion_worker.runtime_identity import patches_identity
        return [{"target": p["target"], "state": p["state"], "installed_sha256": p["installed_sha256"]} for p in patches_identity()]
    except Exception:
        return []


def record(freeze_file=None):
    pins = installed_pins()
    return {"schema": "environment-reference-v1", "pins": pins, "meralion_artifact_sha256": artifact_hash_of(), "torch": torch_facts(), "audio": audio_facts(pins), "patches": patch_states(),
            "informational": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(), "system": platform.system(),
                              "freeze_text_sha256": hashlib.sha256(Path(freeze_file).read_bytes()).hexdigest() if freeze_file else None, "never_compared": True},
            "ignored_packages": list(IGNORED)}


def compare(reference, current, package_manifest, expect_patches, verify_exit):
    """-> a result with `passed` and every difference. `current` has the same shape as record()'s output."""
    differences = []
    ref_pins = {k: v for k, v in reference["pins"].items() if k not in IGNORED}
    cur_pins = {k: v for k, v in current["pins"].items() if k not in IGNORED}
    for name in sorted(set(ref_pins) - set(cur_pins)):
        differences.append(f"missing package {name} (reference {ref_pins[name]})")
    for name in sorted(set(cur_pins) - set(ref_pins)):
        differences.append(f"extra package {name} {cur_pins[name]}")
    for name in sorted(set(ref_pins) & set(cur_pins)):
        if ref_pins[name] != cur_pins[name]:
            differences.append(f"package {name}: reference {ref_pins[name]}, here {cur_pins[name]}")
    manifest_hash = ((package_manifest or {}).get("artifact") or {}).get("sha256")
    if current["meralion_artifact_sha256"] != reference.get("meralion_artifact_sha256"):
        differences.append(f"the {PACKAGE} wheel hash {current['meralion_artifact_sha256']!r} differs from the reference {reference.get('meralion_artifact_sha256')!r}")
    if manifest_hash and current["meralion_artifact_sha256"] != manifest_hash:
        differences.append(f"the {PACKAGE} wheel hash {current['meralion_artifact_sha256']!r} is not the verified package manifest's {manifest_hash!r}")
    if not manifest_hash:
        differences.append("the package manifest names no artifact sha256")
    for key in ("version", "cuda"):
        if current["torch"].get(key) != reference["torch"].get(key):
            differences.append(f"torch {key}: reference {reference['torch'].get(key)!r}, here {current['torch'].get(key)!r}")
    for key in sorted(set(reference["audio"]) | set(current["audio"])):
        if current["audio"].get(key) != reference["audio"].get(key):
            differences.append(f"audio library {key}: reference {reference['audio'].get(key)!r}, here {current['audio'].get(key)!r}")
    states = {entry["state"] for entry in current["patches"]}
    allowed = {"patched": {"patched"}, "original": {"original", "not_installed"}, "none": {"original", "not_installed"}}[expect_patches]
    if current["patches"] or expect_patches == "patched":
        if not states <= allowed or (expect_patches == "patched" and not states):
            differences.append(f"tracked patch states {sorted(states)} do not match the expectation {expect_patches!r}")
    here, there = current.get("informational") or {}, reference.get("informational") or {}
    minor = lambda version: ".".join(str(version).split(".")[:2]) if version else None            # the PATCH version is never compared (3.12.3 and 3.12.8 are both fine)
    if here.get("system") != TARGET_SYSTEM:
        differences.append(f"the operating system is {here.get('system')!r}, the deployment target is {TARGET_SYSTEM!r}")
    if here.get("machine") != TARGET_MACHINE:
        differences.append(f"the machine architecture is {here.get('machine')!r}, the deployment target is {TARGET_MACHINE!r}")
    if minor(here.get("python")) != TARGET_PYTHON_MINOR:
        differences.append(f"Python is {here.get('python')!r}, the target is {TARGET_PYTHON_MINOR}.x (the pinned wheels are cp312)")
    if minor(here.get("python")) != minor(there.get("python")):
        differences.append(f"Python {here.get('python')!r} is not the same minor version as the reference's {there.get('python')!r}")
    if verify_exit != 0:
        differences.append(f"verify-environment exited {verify_exit}: the installed package files are not shown to equal the verified manifest (identical pins are not enough)")
    return {"passed": not differences, "differences": differences, "compared": ["package names and pinned versions", "wheel artifact hash", "torch version and CUDA build", "audio libraries", "tracked patch states", "deployment target (Linux, x86_64, Python 3.12.x, the reference's Python minor version)",
                                                                                 "verify-environment exit"],
            "never_compared": ["paths", "host", "kernel", "python patch version", "freeze text"], "ignored_packages": list(IGNORED),
            "informational": {"reference": reference.get("informational"), "here": current.get("informational")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--record", metavar="OUT")
    mode.add_argument("--check", metavar="REFERENCE")
    parser.add_argument("--freeze-file")
    parser.add_argument("--package-manifest")
    parser.add_argument("--verify-environment-exit", type=int)
    parser.add_argument("--expect-patches", choices=("none", "original", "patched"), default="none")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.record:
        Path(args.record).write_text(json.dumps(record(args.freeze_file), indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"environment reference written to {args.record}")
        return 0
    if args.verify_environment_exit is None or not args.package_manifest or not args.out:
        parser.error("--check needs --package-manifest, --verify-environment-exit and --out")
    reference = json.loads(Path(args.check).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.package_manifest).read_text(encoding="utf-8"))
    result = compare(reference, record(), manifest, args.expect_patches, args.verify_environment_exit)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": result["passed"], "differences": result["differences"]}, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
