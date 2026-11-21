"""Fingerprint what the worker's own audio loader produces (REV 3.2 Amendment 4/7; run on the cluster login node in .venv-vllm for the reference, on the EC2 host for the check).

    python audio_fingerprint.py --jobs jobs.json [--calls CALL_064 CALL_103 ...] --out fingerprint.json
    python audio_fingerprint.py --jobs jobs.json [--calls ...] --check reference.json --out check.json

Per recording: the number of samples at 16 kHz and the sha256 of the loaded float32 array, loaded through meralion_worker.audio.load_recording (soundfile and the package's resampler). A different value
means a different decoder or resampler, therefore different window bounds: a HARD STOP for any comparison of T1 results with the cluster's. Reads audio, writes only the fingerprint.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def fingerprint_recording(recording):
    import numpy as np
    samples = np.ascontiguousarray(recording.get(0, recording.num_samples), dtype=np.float32)
    return {"num_samples": int(recording.num_samples), "sha256": hashlib.sha256(samples.tobytes()).hexdigest()}


def fingerprint_jobs(jobs_path, calls=None, loader=None):
    from meralion_worker import audio
    loader = loader or audio.load_recording
    jobs = json.loads(Path(jobs_path).read_text(encoding="utf-8"))["jobs"]
    wanted = list(calls) if calls else [job["job_name"] for job in jobs]
    by_name = {job["job_name"]: job["input"] for job in jobs}
    missing = [call for call in wanted if call not in by_name]
    if missing:
        raise SystemExit(f"not in the jobs file: {missing}")
    return {call: fingerprint_recording(loader(call, by_name[call])) for call in wanted}


def compare(reference, found):
    """-> (passed, differences). A recording missing on either side is a difference."""
    differences = []
    for call in sorted(set(reference) | set(found)):
        if call not in found:
            differences.append(f"{call}: not fingerprinted here")
        elif call not in reference:
            differences.append(f"{call}: not in the reference")
        elif reference[call] != found[call]:
            differences.append(f"{call}: reference {reference[call]}, here {found[call]}")
    return not differences, differences


def library_versions():
    from importlib import metadata
    versions = {}
    for name in ("soundfile", "librosa", "soxr", "numpy", "scipy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        import soundfile
        versions["libsndfile"] = soundfile.__libsndfile_version__
    except Exception:
        versions["libsndfile"] = None
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--calls", nargs="*")
    parser.add_argument("--check")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    found = fingerprint_jobs(args.jobs, args.calls)
    result = {"schema": "audio-fingerprint-v1", "recordings": found, "libraries": library_versions()}
    exit_code = 0
    if args.check:
        reference = json.loads(Path(args.check).read_text(encoding="utf-8"))
        passed, differences = compare(reference["recordings"], found)
        result.update(passed=passed, differences=differences, reference_libraries=reference.get("libraries"))
        exit_code = 0 if passed else 1
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"recordings": len(found), "passed": result.get("passed"), "differences": result.get("differences")}, indent=1))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
