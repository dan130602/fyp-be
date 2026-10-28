"""Development-use registry: which calls were used for prompt tuning, qualification, diagnosis or targeted debugging.

This is separate from the provider-exposure ledger (`authoring/exposure_log.json`), which records live API usage for all 250 calls and
cannot define development use (otherwise no held-out set could exist). The ledger is left unchanged; it is cited here only as
historical provenance. Routine generation and independent QA under a fixed procedure do not by themselves lock a call.

`build_registry` reads the LOCAL-ONLY authoring provenance (gitignored) to embed attempt histories as evidence, so the committed
registry stays checkable on any clone: `validate_registry` needs nothing but the registry and the committed call specs.
"""

import json
from pathlib import Path

from fyp_evaluation.planning.outputs import sha256

from . import REGISTRY_FILE, SPLIT_DIR, SplitError

SCHEMA_VERSION = "1.0"

USER_GIVEN = (
    "CALL_002", "CALL_003", "CALL_004", "CALL_010", "CALL_014", "CALL_023", "CALL_024", "CALL_027", "CALL_028", "CALL_030",
    "CALL_034", "CALL_035", "CALL_036", "CALL_043", "CALL_044", "CALL_047", "CALL_059", "CALL_064", "CALL_102", "CALL_103",
    "CALL_112", "CALL_113", "CALL_122", "CALL_152", "CALL_155", "CALL_156", "CALL_165", "CALL_178", "CALL_203", "CALL_208",
    "CALL_209", "CALL_227", "CALL_250",
)

PILOT_BATCHES = {"PILOT_1", "PILOT_2", "PILOT_2_G6V5", "PILOT_CODEX_1"}

# Calls outside the pilot batches, with the diagnosed reason (user_given first, then audit_derived).
DIAGNOSED = {
    "CALL_035": ("user_given", "DIAGNOSED_REPAIR", "G3 variant-language contradiction (EA_085 vs skeleton EV_012 required_language); G3 reopened with versioned "
                 "G3 v3 variant-purpose guidance, then G4 and G5 rebuilt; the original G5 was INVALID (SCENARIO_CONTRADICTION)."),
    "CALL_059": ("user_given", "DIAGNOSED_REPAIR", "G4 A001 rejected for an unknown [[Q]] placeholder; diagnosed and corrected through the A002 retry."),
    "CALL_122": ("user_given", "DIAGNOSED_REPAIR", "G5 held the call: the G4 readback needs two or more transaction fields (an ID is not a transaction field); "
                 "repaired with versioned G4 v4 guidance."),
    "CALL_156": ("user_given", "DIAGNOSED_REPAIR", "G5 held the call: G4 C07 needs a product/action rationale, not an execution-method rationale; repaired with "
                 "versioned G4 v4 guidance."),
    "CALL_203": ("user_given", "DIAGNOSED_REPAIR", "G5 held the call for the same C07 execution-method-only rationale as CALL_156; repaired with G4 v4."),
    "CALL_208": ("user_given", "DIAGNOSED_REPAIR", "G5 held the call for the same one-transaction-field readback as CALL_122; repaired with G4 v4."),
    "CALL_155": ("user_given", "DIAGNOSED_REGENERATION", "G6 A001 rejected for joined event markers (marker-spacing rule); diagnosed and regenerated once (A002)."),
    "CALL_250": ("user_given", "DIAGNOSED_REGENERATION", "G6 A001 rejected for the same marker-spacing defect as CALL_155; regenerated once (A002)."),
    "CALL_149": ("audit_derived", "DIAGNOSED_REPAIR", "G4 A001 rejected for the same unknown [[Q]] placeholder defect as CALL_059; together they motivated the G4 v3 "
                 "prompt, so it is locked for consistency with CALL_059."),
    "CALL_090": ("audit_derived", "DIAGNOSED_REGENERATION", "G7 found an unplanned refusal/escalation in supporting speech (CROSS_CHECK_INTERFERENCE); the G6 "
                 "dialogue was regenerated and verified again."),
    "CALL_147": ("audit_derived", "DIAGNOSED_REGENERATION", "G7 found an unplanned compliance escalation in supporting speech (CROSS_CHECK_INTERFERENCE); the G6 "
                 "dialogue was regenerated and verified again."),
    "CALL_071": ("audit_derived", "DIAGNOSED_REGENERATION", "G7 A001 rejected on report structure; the G6 dialogue was regenerated on the user's instruction and "
                 "verified again."),
    "CALL_134": ("audit_derived", "VERIFIER_DEBUGGING", "Three G7 reports were truncated at the output limit; this call drove the G7 output-limit and attempt-budget "
                 "changes and its G6 dialogue was regenerated."),
    "CALL_163": ("audit_derived", "VERIFIER_DEBUGGING", "G7 A001 rejected (quote not a substring of the cited turn, PRESENCE check marked MATCH without its own "
                 "evidence event); diagnosed against the dialogue before the retry."),
    "CALL_198": ("audit_derived", "VERIFIER_DEBUGGING", "Two G7 reports rejected for over-long quotes and an edge-case MATCH citing the wrong events; diagnosed "
                 "against the dialogue; still without an accepted G7 report."),
    "CALL_218": ("audit_derived", "VERIFIER_DEBUGGING", "Two G7 reports rejected for a quote that is not a substring of the cited turn; diagnosed against the "
                 "dialogue; still without an accepted G7 report."),
}

PILOT_REASON = {
    "PILOT_1": "Initial live qualification pilot (one call per language profile) used to qualify G3-G7 behaviour; Codex qualification also reviewed these calls.",
    "PILOT_2": "Expansion pilot used for G6 prompt v3 calibration, the v5 A/B regeneration, and the G7 authoring-conformance scope diagnosis.",
    "PILOT_CODEX_1": "Codex-harness qualification pilot (two LONG, three SHORT, all four language profiles) before the 225-call rollout.",
}
PILOT_ORDER = ("PILOT_1", "PILOT_2", "PILOT_CODEX_1")  # PILOT_2_G6V5 covers the same calls as PILOT_2

_PROVENANCE = {"G3": "call_config_provenance", "G4": "check_plan_provenance", "G5": "plan_verification_provenance",
               "G6": "dialogue_provenance", "G7": "dialogue_verification_provenance"}


def _history(authoring, call_id):
    """{stage: 'A001:REJECTED A002:ACCEPTED'} for stages with more than one attempt or a reopen (local-only provenance)."""
    out = {}
    for stage, folder in _PROVENANCE.items():
        path = authoring / folder / f"{call_id}.json"
        if not path.is_file():
            continue
        attempts = json.loads(path.read_text(encoding="utf-8"))["attempts"]
        if len(attempts) > 1 or any(a.get("reopen") for a in attempts):
            out[stage] = " ".join(f"{a['attempt_id']}:{a['state']}" for a in attempts)
    return out


def _count_attempts(authoring):
    total = 0
    for folder in _PROVENANCE.values():
        for path in (authoring / folder).glob("CALL_*.json"):
            total += len(json.loads(path.read_text(encoding="utf-8"))["attempts"])
    return total


def build_registry(dataset_root, dataset_id):
    """The registry document. Reads local-only provenance for evidence; run on the machine that holds the authoring chain."""
    root = Path(dataset_root)
    authoring = root / "authoring"
    allowlist = json.loads((authoring / "live_allowlist.json").read_text(encoding="utf-8"))["batches"]
    batch_of = {}
    for batch_id in (*PILOT_ORDER, "PILOT_2_G6V5"):
        for call in allowlist[batch_id]["call_ids"]:
            batch_of.setdefault(call, []).append(batch_id)
    locks = {}
    for call, batches in batch_of.items():
        first = next(b for b in PILOT_ORDER if b in batches)
        locks[call] = {"category": "PILOT_PROMPT_TUNING", "source": "user_given", "reasons": [PILOT_REASON[first]],
                       "allowlist_batches": sorted(batches)}
    for call, (source, category, reason) in DIAGNOSED.items():
        entry = locks.setdefault(call, {"allowlist_batches": [], "reasons": [], "category": category, "source": source})
        entry["reasons"].append(reason)  # a pilot call that was also diagnosed keeps its pilot category and gains the reason
    for call in USER_GIVEN:
        if call not in locks:
            raise SplitError(f"{call}: a user-given lock has no recorded reason")
        locks[call]["source"] = "user_given"
    entries = []
    for call in sorted(locks):
        entry = locks[call]
        entries.append({"call_id": call, "status": "DEVELOPMENT_LOCK", "category": entry["category"], "source": entry["source"],
                        "reasons": entry["reasons"],
                        "evidence": {"allowlist_batches": entry["allowlist_batches"], "attempt_history": _history(authoring, call),
                                     "local_sources": [".llm_audit/ROLLOUT_CODEX_1/progress.json", "authoring/*_provenance/" + call + ".json"]}})
    # routine, not locked: grouped by identical non-clean history (transport/capacity or batch rejection retries only)
    patterns = {}
    ledger_path = authoring / "exposure_log.json"
    for path in sorted((authoring / "call_specs").glob("CALL_*.json")):
        call = path.stem
        if call in locks:
            continue
        history = _history(authoring, call)
        if history:
            patterns.setdefault(json.dumps(history, sort_keys=True), []).append(call)
    routine = [{"pattern": json.loads(k), "calls": v, "why_not_locked": "routine retry of a transport, capacity or batch-level rejection under a "
                "fixed procedure; no prompt, schema or settings change was derived from the call"} for k, v in sorted(patterns.items())]
    uncertain = []
    for folder in _PROVENANCE.values():
        for path in sorted((authoring / folder).glob("CALL_*.json")):
            for a in json.loads(path.read_text(encoding="utf-8"))["attempts"]:
                if a.get("possibly_billed") or a.get("ambiguous"):
                    uncertain.append({"call_id": path.stem, "stage": a["stage"] if "stage" in a else folder, "attempt_id": a["attempt_id"],
                                      "locked": path.stem in locks, "note": "attempt outcome was ambiguous or possibly billed; this is "
                                      "billing uncertainty, not development use"})
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    return {
        "schema_version": SCHEMA_VERSION, "registry_version": "development_exposure.v1", "dataset_id": dataset_id,
        "method": "retrospective: derived after generation from allowlisted pilot batches and diagnosed repair/regeneration history; "
                  "not frozen before generation, so it does not support a stronger independence claim",
        "rules": ["a locked call can only be added, never removed to improve balance",
                  "routine generation, transport retries and independent QA under a fixed procedure do not lock a call",
                  "uncertain exposure is reported and not silently treated as clean"],
        "exposure_ledger": {"path": "authoring/exposure_log.json", "sha256": sha256(ledger_path.read_bytes()),
                            "live_attempts": ledger["summary"]["live_attempts"] if "summary" in ledger and "live_attempts" in ledger["summary"]
                            else ledger.get("live_attempts"), "exposed_calls": len(ledger["exposed_calls"]),
                            "provenance_attempts_now": _count_attempts(authoring), "status": "STALE_HISTORICAL_ONLY",
                            "note": "lists all 250 calls as exposed (live API usage) and ends before the 225-call rollout finished; unchanged, and not "
                                    "used to define development use"},
        "user_given_locks": sorted(USER_GIVEN),
        "audit_derived_locks": sorted(c for c, v in DIAGNOSED.items() if v[0] == "audit_derived"),
        "locks": entries, "routine_not_locked": routine, "uncertain_exposure": uncertain,
        "groups": {"genuine_groups": [], "audit": "no explicit link/pair/base-call field exists; 6 pairs share an order tuple and 5 groups share "
                   "edge-case parameters but differ in language, edge cases and verdicts (coincidences of small draw domains), so none is a leakage group",
                   "coincident_order_pairs": [["CALL_007", "CALL_166"], ["CALL_014", "CALL_022"], ["CALL_047", "CALL_222"], ["CALL_064", "CALL_190"],
                                              ["CALL_111", "CALL_219"], ["CALL_151", "CALL_174"]],
                   "coincident_edge_parameter_groups": [["CALL_026", "CALL_192", "CALL_212"], ["CALL_064", "CALL_095"], ["CALL_158", "CALL_197"],
                                                        ["CALL_171", "CALL_243"], ["CALL_184", "CALL_226"]]},
    }


def validate_registry(registry, features=None):
    """Errors for a registry document (offline; needs no ignored files)."""
    errors = []
    locks = {e["call_id"]: e for e in registry.get("locks", [])}
    if len(locks) != len(registry.get("locks", [])):
        errors.append("duplicate lock entries")
    for call in USER_GIVEN:
        if call not in locks:
            errors.append(f"{call}: a user-given lock was removed")
        elif locks[call]["source"] != "user_given":
            errors.append(f"{call}: a user-given lock lost its user_given source")
    if sorted(registry.get("user_given_locks", [])) != sorted(USER_GIVEN):
        errors.append("user_given_locks differs from the 33 conservative locks")
    for call, entry in locks.items():
        if entry.get("status") != "DEVELOPMENT_LOCK" or not entry.get("reasons"):
            errors.append(f"{call}: lock needs status DEVELOPMENT_LOCK and at least one reason")
        if features is not None and call not in features:
            errors.append(f"{call}: lock is not a call of the release")
    derived = {c for c, e in locks.items() if e["source"] == "audit_derived"}
    if derived != set(registry.get("audit_derived_locks", [])):
        errors.append("audit_derived_locks differs from the lock sources")
    return errors


def registry_path(dataset_root):
    return Path(dataset_root) / SPLIT_DIR / REGISTRY_FILE
