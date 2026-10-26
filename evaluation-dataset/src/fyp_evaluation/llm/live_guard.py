"""Technical controls for paid runs on a multilingual release: the live allowlist and the code-state record.

Supplying call IDs never authorises a paid run. A committed allowlist (`authoring/live_allowlist.json`) names each approved
batch (its stages and call IDs); it is added only after a human has approved that batch and its budget. A paid run on a
multilingual release must select calls that all belong to one allowlisted batch for its stage. Runs on fake transports
(tests, offline work) and on the historical release are never affected.

Every live attempt also records the code commit and any uncommitted code, so a live exposure is reproducible. Unrelated
work (data outputs, runtime artifacts, other directories) never blocks a run and is never stashed or discarded.
"""

import subprocess
from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text

ALLOWLIST = "authoring/live_allowlist.json"
# Relative to the evaluation-dataset project root. `scripts` counts only the top-level pipeline command-line scripts, never the
# subdirectories (for example the unrelated GPU TTS tooling), so unrelated work never blocks a run.
CODE_PATHS = ("src", "prompts", "schemas", ":(glob)scripts/*.py")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FAKE_MODULES = ("fake_transport", "fake_gemini_transport")


def is_fake_transport(transport):
    """True for the in-memory test transports (live and offline runs are recorded differently)."""
    return transport is None or type(transport).__module__.rsplit(".", 1)[-1] in FAKE_MODULES


def code_state(project_root=PROJECT_ROOT):
    """{"code_commit": sha or None, "code_dirty_paths": [paths]} for the code directories (git status, porcelain)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=project_root, capture_output=True, text=True, timeout=60, check=False)

    try:
        head = git("rev-parse", "HEAD")
        status = git("status", "--porcelain", "--untracked-files=normal", "--", *CODE_PATHS)
    except (OSError, subprocess.SubprocessError):
        return {"code_commit": None, "code_dirty_paths": []}
    if head.returncode != 0 or status.returncode != 0:
        return {"code_commit": None, "code_dirty_paths": []}
    dirty = sorted(line[3:].strip().strip('"') for line in status.stdout.splitlines() if line.strip())
    return {"code_commit": head.stdout.strip(), "code_dirty_paths": dirty}


def load_allowlist(root):
    path = Path(root) / ALLOWLIST
    if not path.is_file():
        return {}
    data = parse_json_text(path.read_text(encoding="utf-8"))
    batches = data.get("batches")
    if not isinstance(batches, dict):
        raise ValueError(f"{ALLOWLIST}: needs a 'batches' object")
    return batches


def authorised_batch(root, stage, call_ids):
    """The ID of the one allowlisted batch covering `stage` and every call, else a ValueError naming what is missing."""
    batches = load_allowlist(root)
    if not call_ids:
        raise ValueError("no calls selected")
    matching = [batch_id for batch_id, batch in sorted(batches.items())
                if stage in batch.get("stages", []) and set(call_ids) <= set(batch.get("call_ids", []))]
    if not matching:
        raise ValueError(f"calls {', '.join(sorted(call_ids))} are not all in one {stage} batch of {ALLOWLIST}; a paid run on a "
                         "multilingual release needs a human-approved allowlisted batch")
    return matching[0]
