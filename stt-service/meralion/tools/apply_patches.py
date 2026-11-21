"""Check, apply or revert the tracked compatibility patches of the vLLM environment (REV 3.0 A3). Standard library only.

    python apply_patches.py --check  [--expect patched|original] [--env SITE_PACKAGES] [--manifest patches/manifest.json]
    python apply_patches.py --apply  --confirm-env-idle [--reviewed-job ID ...] [--env ...]
    python apply_patches.py --revert --confirm-env-idle [--reviewed-job ID ...] [--env ...]

The manifest (patches/manifest.json) names, per patch, the target file inside site-packages, the vLLM version it is for and the sha256 of the original file, of the patched file and of the patch.
meralion_worker.runtime_identity.patches_identity() reads the same manifest, so every document records whether the patch was applied.

Rules: an unknown vLLM version or an original file hash this tool does not know is refused (nothing is written); the original is copied to <target>.orig once; the patched file is written to a temporary
file in the SAME directory and moved with os.replace, so a concurrent import sees the whole old or the whole new file; the result is verified against the manifest's patched hash.
--apply AND --revert (REV 4.1 A2: ONE idle policy, checked only when a write would actually happen) refuse unless --confirm-env-idle is given AND squeue (any state: it lists only jobs that are not
finished) shows no job whose name starts with w0- AND every OTHER listed job (a notebook, a manual sbatch) is named with --reviewed-job <id>. A reviewed id that is not in the queue is refused (a stale or
mistyped acknowledgement must not hide a job). --reviewed-job is a HUMAN acknowledgement that the job does not use this environment: this tool cannot establish that, and it never stops or changes a job.
It is applied only on request, never from a job.
"""
import argparse
import getpass
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "patches" / "manifest.json"
JOB_NAME_PREFIX = "w0-"
SQUEUE_FORMAT = "%i|%j|%T"


class PatchError(RuntimeError):
    """The patch cannot be applied, checked or reverted safely (nothing was written)."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ---- a minimal unified-diff applier (context-checked; the patches of this project are small and have no renames or binary parts) ----

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_hunks(patch_text):
    hunks, current = [], None
    for line in patch_text.split("\n"):
        header = HUNK.match(line)
        if header:
            current = {"old_start": int(header.group(1)), "old_count": int(header.group(2) or 1), "lines": []}
            hunks.append(current)
        elif current is not None and line[:1] in (" ", "+", "-"):
            current["lines"].append((line[0], line[1:]))
        elif current is not None and line.startswith("\\"):          # "\ No newline at end of file"
            continue
        elif line.startswith(("--- ", "+++ ", "diff ", "index ")) or line == "":
            continue
    if not hunks:
        raise PatchError("the patch has no hunks")
    return hunks


def apply_unified(original_text, patch_text):
    """Apply the hunks to `original_text` (LF line endings) and return the new text. Every context and removed line must match exactly."""
    lines = original_text.split("\n")
    offset = 0
    for hunk in parse_hunks(patch_text):
        old = [text for kind, text in hunk["lines"] if kind in (" ", "-")]
        new = [text for kind, text in hunk["lines"] if kind in (" ", "+")]
        if len(old) != hunk["old_count"]:
            raise PatchError(f"hunk at line {hunk['old_start']}: its header says {hunk['old_count']} old lines but it carries {len(old)}")
        start = hunk["old_start"] - 1 + offset
        if lines[start:start + len(old)] != old:
            raise PatchError(f"hunk at line {hunk['old_start']} does not match the file (the file is not the one the patch was made for)")
        lines[start:start + len(old)] = new
        offset += len(new) - len(old)
    return "\n".join(lines)


# ---- the manifest and the installed environment ----

def load_manifest(path=None):
    path = Path(path) if path else DEFAULT_MANIFEST
    if not path.is_file():
        raise PatchError(f"patch manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("patches")
    if not entries:
        raise PatchError("the manifest lists no patches")
    return path, entries


def default_env_root():
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise PatchError("vllm is not importable here; pass --env <site-packages>")
    return Path(list(spec.submodule_search_locations)[0]).parent


def installed_version(env_root, package="vllm"):
    """The installed version, read from the dist-info METADATA without importing the package."""
    for metadata in sorted(Path(env_root).glob(f"{package}-*.dist-info/METADATA")):
        match = re.search(r"^Version: (\S+)", metadata.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
        if match:
            return match.group(1)
    return None


def inspect_entry(env_root, entry, patches_dir):
    target = Path(env_root) / entry["target"]
    patch_file = Path(patches_dir) / entry["patch_file"]
    digest = sha256_bytes(target.read_bytes()) if target.is_file() else None
    state = "missing" if digest is None else "original" if digest == entry["original_sha256"] else "patched" if digest == entry["patched_sha256"] else "unknown"
    patch_digest = sha256_bytes(patch_file.read_bytes()) if patch_file.is_file() else None
    return {"target": entry["target"], "target_version": entry["target_version"], "installed_version": installed_version(env_root), "state": state, "installed_sha256": digest,
            "original_sha256": entry["original_sha256"], "patched_sha256": entry["patched_sha256"], "patch_sha256": entry["patch_sha256"], "patch_file_sha256": patch_digest,
            "patch_file_matches_manifest": patch_digest == entry["patch_sha256"], "backup_exists": Path(str(target) + ".orig").is_file()}


def parse_queue(squeue_text):
    """The jobs in `squeue -h -o '%i|%j|%T'` output: [{"id", "name", "state"}]. The name may contain spaces or '|' (the id is before the first '|', the state after the last one);
    a malformed line is an error, never skipped."""
    jobs = []
    for line in squeue_text.splitlines():
        if not line.strip():
            continue
        parsed = None
        if line.count("|") >= 2:
            job_id, rest = line.split("|", 1)
            name, state = rest.rsplit("|", 1)
            if job_id.strip() and state.strip():
                parsed = {"id": job_id.strip(), "name": name.strip(), "state": state.strip()}
        if parsed is None:
            raise PatchError(f"refused: cannot parse the squeue line {line!r}; the environment cannot be shown to be idle")
        jobs.append(parsed)
    return jobs


def run_squeue():
    result = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", SQUEUE_FORMAT], capture_output=True, text=True)
    if result.returncode != 0:
        raise PatchError(f"squeue failed ({result.returncode}): {result.stderr.strip()}; the environment cannot be shown to be idle")
    return result.stdout


def require_idle(squeue, reviewed_jobs, confirm):
    """The ONE idle policy of --apply and --revert (REV 4.1 A2). Returns the evidence {"queue", "reviewed_jobs", "note"}; raises PatchError (nothing may be written) unless: the caller confirmed;
    the queue could be read and parsed; no listed job (in ANY state: squeue lists only jobs that are not finished) is named w0-*; every other listed job is named in `reviewed_jobs`; and every
    reviewed id is in the queue. A reviewed job is a human acknowledgement, not proof that the environment is unused."""
    if not confirm:
        raise PatchError("refused: --confirm-env-idle is required (nothing else may be using this environment, and a notebook job cannot be checked by this tool)")
    queue = parse_queue(squeue())
    reviewed = [str(job) for job in (reviewed_jobs or ())]
    w0 = [job for job in queue if job["name"].startswith(JOB_NAME_PREFIX)]
    if w0:
        raise PatchError("refused: w0- jobs are in the queue (a job in any state blocks): " + "; ".join(f"{j['id']} {j['name']} {j['state']}" for j in w0))
    queued_ids = {job["id"] for job in queue}
    stale = [job for job in reviewed if job not in queued_ids]
    if stale:
        raise PatchError(f"refused: --reviewed-job {', '.join(stale)} is not in the queue (a stale or mistyped acknowledgement must not hide a job); queue: {sorted(queued_ids)}")
    unreviewed = [job for job in queue if job["id"] not in reviewed]
    if unreviewed:
        raise PatchError("refused: these queued jobs have not been reviewed (name each with --reviewed-job <id> once you have checked that it does not use this environment): "
                         + "; ".join(f"{j['id']} {j['name']} {j['state']}" for j in unreviewed))
    return {"queue": queue, "reviewed_jobs": reviewed,
            "note": "reviewed jobs are a human acknowledgement; this tool cannot establish that a reviewed job does not use the environment, and it never stops or changes a job"}


def write_replace(target, data):
    """Write to a temporary file in the same directory, then os.replace: a reader sees the whole old or the whole new file."""
    target = Path(target)
    temporary = target.with_name(target.name + ".tmp-apply")
    temporary.write_bytes(data)
    os.replace(temporary, target)


def apply_entry(env_root, entry, patches_dir, *, confirm_env_idle, reviewed_jobs=(), squeue=run_squeue):
    info = inspect_entry(env_root, entry, patches_dir)
    if info["installed_version"] != entry["target_version"]:
        raise PatchError(f"refused: the patch is for vllm {entry['target_version']}, installed is {info['installed_version']}")
    if not info["patch_file_matches_manifest"]:
        raise PatchError("refused: the patch file's sha256 differs from the manifest")
    if info["state"] == "patched":
        return {**info, "action": "already patched"}
    if info["state"] != "original":
        raise PatchError(f"refused: the installed file is in state {info['state']!r} (sha256 {info['installed_sha256']}); only the recorded original can be patched")
    idle = require_idle(squeue, reviewed_jobs, confirm_env_idle)
    target = Path(env_root) / entry["target"]
    original = target.read_bytes()
    text = original.decode("utf-8")
    if "\r" in text:
        raise PatchError("refused: the target has CRLF line endings; the patch is made for LF")
    patched = apply_unified(text, (Path(patches_dir) / entry["patch_file"]).read_text(encoding="utf-8")).encode("utf-8")
    if sha256_bytes(patched) != entry["patched_sha256"]:
        raise PatchError("refused: the patched result does not match the manifest's patched sha256 (nothing was written)")
    backup = Path(str(target) + ".orig")
    if not backup.exists():
        backup.write_bytes(original)
    write_replace(target, patched)
    after = inspect_entry(env_root, entry, patches_dir)
    if after["state"] != "patched":
        raise PatchError(f"the file is in state {after['state']!r} after writing")
    return {**after, "action": "applied", "idle_check": idle}


def revert_entry(env_root, entry, patches_dir, *, confirm_env_idle, reviewed_jobs=(), squeue=run_squeue):
    info = inspect_entry(env_root, entry, patches_dir)
    if info["state"] == "original":
        return {**info, "action": "already original"}
    if info["state"] != "patched":
        raise PatchError(f"refused: the installed file is in state {info['state']!r}; only the recorded patched file can be reverted")
    target = Path(env_root) / entry["target"]
    backup = Path(str(target) + ".orig")
    if not backup.is_file() or sha256_bytes(backup.read_bytes()) != entry["original_sha256"]:
        raise PatchError("refused: the backup .orig is missing or is not the recorded original")
    idle = require_idle(squeue, reviewed_jobs, confirm_env_idle)
    write_replace(target, backup.read_bytes())
    return {**inspect_entry(env_root, entry, patches_dir), "action": "reverted", "idle_check": idle}


def check_entry(env_root, entry, patches_dir, expect):
    info = inspect_entry(env_root, entry, patches_dir)
    problems = []
    if info["installed_version"] != entry["target_version"]:
        problems.append(f"installed vllm is {info['installed_version']}, the patch is for {entry['target_version']}")
    if not info["patch_file_matches_manifest"]:
        problems.append("the patch file's sha256 differs from the manifest")
    if info["state"] != expect:
        problems.append(f"the installed file is {info['state']!r}, expected {expect!r}")
    return {**info, "expected_state": expect, "problems": problems, "ok": not problems}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--apply", action="store_true")
    action.add_argument("--revert", action="store_true")
    parser.add_argument("--expect", choices=("patched", "original"), default="patched")
    parser.add_argument("--confirm-env-idle", action="store_true")
    parser.add_argument("--reviewed-job", action="append", default=[], metavar="ID",
                        help="a queued job you have checked does not use this environment (repeatable; a human acknowledgement, not proof)")
    parser.add_argument("--env", type=Path, default=None, help="the site-packages root (default: where vllm is importable)")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        manifest_path, entries = load_manifest(args.manifest)
        env_root = args.env or default_env_root()
        results = []
        for entry in entries:
            if args.check:
                results.append(check_entry(env_root, entry, manifest_path.parent, args.expect))
            elif args.apply:
                results.append(apply_entry(env_root, entry, manifest_path.parent, confirm_env_idle=args.confirm_env_idle, reviewed_jobs=args.reviewed_job, squeue=run_squeue))
            else:
                results.append(revert_entry(env_root, entry, manifest_path.parent, confirm_env_idle=args.confirm_env_idle, reviewed_jobs=args.reviewed_job, squeue=run_squeue))
    except PatchError as error:
        print(f"apply_patches: {error}", file=sys.stderr)
        return 3
    print(json.dumps({"env": str(env_root), "manifest": str(manifest_path), "results": results}, indent=1))
    return 0 if all(result.get("ok", True) for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
