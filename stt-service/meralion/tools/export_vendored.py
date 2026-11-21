"""Export the frozen runtime sources from Git as exact bytes and record their identities.

    python tools/export_vendored.py                               # run from anywhere inside the repository

Every file is pinned to its OWN commit (PINS below), so a file can be promoted without moving the others:
  overlap.py, retry_plan.py, retry_merge.py   2f982b2   the frozen merge and retry logic (unchanged by the amt1 promotion)
  formatting.py                               6feb3fa   the formatter that adds amount-v1 next to percentages-v2 (Stage 2A); the blob adds 130 lines to the 2f982b2 one and deletes none
The bytes come from `git cat-file blob <commit>:<path>` captured as bytes and written with Path.write_bytes: no shell redirection, no newline or encoding
translation. After writing, the git blob id is recomputed from the written bytes and must equal `git rev-parse <commit>:<path>`; otherwise nothing is recorded.
Writes meralion_worker/vendored/<name>.py and meralion_worker/vendored/SOURCES.json (schema v2: each file records its own commit, blob id, sha256 and size).
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
VENDORED = PACKAGE / "meralion_worker" / "vendored"
# source path inside the repository -> (file name in vendored/, the commit that file is pinned to)
PINS = {
    "stt-experiments/meralion-runner/overlap.py": ("overlap.py", "2f982b2"),
    "stt-experiments/meralion-runner/retry_plan.py": ("retry_plan.py", "2f982b2"),
    "stt-experiments/meralion-runner/retry_merge.py": ("retry_merge.py", "2f982b2"),
    "stt-experiments/meralion/formatting.py": ("formatting.py", "6feb3fa"),
}


def git(*args, cwd):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout


def blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    root = Path(git("rev-parse", "--show-toplevel", cwd=PACKAGE).decode().strip())
    records, staged = {}, {}
    for source, (name, pin) in PINS.items():
        commit = git("rev-parse", pin, cwd=root).decode().strip()
        data = git("cat-file", "blob", f"{commit}:{source}", cwd=root)
        expected = git("rev-parse", f"{commit}:{source}", cwd=root).decode().strip()
        actual = blob_id(data)
        if actual != expected:
            raise SystemExit(f"blob id mismatch for {source}: recomputed {actual}, git says {expected}")
        staged[name] = data
        records[name] = {"source_path": source, "commit": commit, "git_blob": expected, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    VENDORED.mkdir(parents=True, exist_ok=True)
    for name, data in staged.items():
        (VENDORED / name).write_bytes(data)
    for name in staged:                                   # read back what was written, not what was intended
        written = (VENDORED / name).read_bytes()
        if blob_id(written) != records[name]["git_blob"]:
            raise SystemExit(f"written file differs from the blob: {name}")
    document = {"schema": "vendored-sources-v2", "files": records}
    (VENDORED / "SOURCES.json").write_bytes((json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
    for name, record in records.items():
        print(f"{name}: commit {record['commit'][:8]} blob {record['git_blob'][:16]} sha256 {record['sha256'][:16]} {record['bytes']} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
