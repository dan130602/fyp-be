"""The exposure ledger: which calls have been sent to a real provider, derived from what the lifecycle already persists.

Nothing here writes during a run. Every attempt is stored in the (local, gitignored) audit store as `PREPARED`, with its
`exposure` record, *before* anything is sent, and its terminal summary is later committed in the stage's provenance file.
This module merges both sources, so rejected, interrupted and possibly-billed attempts are never omitted, and writes a
deterministic `authoring/exposure_log.json`. Only attempts recorded with `run_kind == "live"` count as exposure; fake-transport
and offline attempts are excluded. The ledger sits outside every hashed G0-G2B input. Exposed calls stay identifiable so a
future development/held-out split can place them in development (no split is implemented here).
"""

from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json

LEDGER = "authoring/exposure_log.json"
FIELDS = ("transport", "state", "possibly_billed", "model", "prompt_template_sha256", "response_schema_sha256", "request_sha256",
          "upstream_sha256", "estimated_cost_usd", "created_at", "finished_at")


def _provenance_attempts(root):
    for path in sorted((Path(root) / "authoring").glob("*_provenance/CALL_*.json")):
        record = parse_json_text(path.read_text(encoding="utf-8"))
        for attempt in record.get("attempts", []):
            yield record["stage"], record["call_id"], attempt, "provenance"


def _audit_attempts(audit_root, dataset_id):
    base = Path(audit_root) / dataset_id
    for path in sorted(base.glob("*/attempts/CALL_*/A*/attempt.json")) if base.is_dir() else []:
        attempt = parse_json_text(path.read_text(encoding="utf-8"))
        yield attempt["stage"], attempt["call_id"], attempt, "audit"


def collect(root, audit_root, dataset_id):
    """Every live attempt (committed or only in the audit store), sorted by stage, call and attempt."""
    merged = {}
    for stage, call_id, attempt, source in [*_provenance_attempts(root), *_audit_attempts(audit_root, dataset_id)]:
        exposure = attempt.get("exposure")
        if not exposure or exposure.get("run_kind") != "live":
            continue
        entry = merged.setdefault((stage, call_id, attempt["attempt_id"]), {"sources": set()})
        entry["sources"].add(source)
        if source == "audit" or "state" not in entry:  # the audit record is the most recent state of the attempt
            entry.update({key: attempt.get(key) for key in FIELDS}, exposure=exposure)
    rows = []
    for (stage, call_id, attempt_id), entry in sorted(merged.items()):
        row = {"stage": stage, "call_id": call_id, "attempt_id": attempt_id, **{key: entry[key] for key in FIELDS}, **entry["exposure"],
               "recorded_in": sorted(entry["sources"])}
        row.pop("run_kind", None)
        rows.append(row)
    return rows


def build(root, audit_root, dataset_id):
    rows = collect(root, audit_root, dataset_id)
    calls = {}
    for row in rows:
        calls.setdefault(row["call_id"], set()).add(row["stage"])
    return {
        "schema_version": "1.0", "dataset_id": dataset_id,
        "note": "Derived by scripts/report_exposure.py from the audit store and committed provenance; live attempts only. "
                "Exposed calls go to the development subset of any future split.",
        "live_attempts": rows,
        "exposed_calls": {call_id: sorted(stages) for call_id, stages in sorted(calls.items())},
        "summary": {"live_attempts": len(rows), "exposed_calls": len(calls),
                    "possibly_billed": sum(1 for row in rows if row["possibly_billed"]),
                    "estimated_cost_usd": round(sum(row["estimated_cost_usd"] or 0 for row in rows), 6)},
    }


def render(root, audit_root, dataset_id):
    return serialize_json(build(root, audit_root, dataset_id))
