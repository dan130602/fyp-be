"""Committed per-call provenance files: attempt summaries, the accepted parsed response and accepted history.

Stage-agnostic: every function takes the `StageSpec` that names the stage and its paths.
"""

from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json

from .fsutil import write_text_atomic

SCHEMA_VERSION = "1.0"
# Attempt fields that stay in the gitignored audit record only.
_AUDIT_ONLY = {"reopen", "blocked", "schema_version", "stage", "dataset_id", "call_id", "attempt_number", "last_error_detail"}


def empty(spec, dataset_id, call_id):
    return {"schema_version": SCHEMA_VERSION, "stage": spec.stage, "dataset_id": dataset_id, "call_id": call_id,
            "attempts": [], "accepted": None, "accepted_history": []}


def read(spec, root, dataset_id, call_id):
    path = Path(root) / spec.provenance_path(call_id)
    if not path.is_file():
        return empty(spec, dataset_id, call_id)
    return parse_json_text(path.read_text(encoding="utf-8"))


def write(spec, root, provenance):
    write_text_atomic(Path(root) / spec.provenance_path(provenance["call_id"]), serialize_json(provenance))


def summary(attempt):
    """Compact committed form of a terminal attempt."""
    result = {key: value for key, value in attempt.items() if key not in _AUDIT_ONLY}
    reopen = attempt.get("reopen")
    result["reopen_reason"] = reopen["reason"] if reopen else None
    return result


def upsert_attempt(provenance, attempt):
    entry = summary(attempt)
    attempts = provenance["attempts"]
    for index, existing in enumerate(attempts):
        if existing["attempt_id"] == attempt["attempt_id"]:
            attempts[index] = entry
            break
    else:
        attempts.append(entry)
        attempts.sort(key=lambda item: item["attempt_id"])
