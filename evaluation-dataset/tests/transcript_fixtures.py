"""ARTIFICIAL ground-truth fixtures for transcript-converter tests only."""

SCHEMA_VERSION = "0.6.0"


def minimal_ground_truth(call_id, turns):
    """Minimal valid ground truth with a single absence-mode check, so no
    evidence-matching is required for it to be schema-valid. `turns` is a
    list of (speaker, text) pairs."""
    return {
        "schema_version": SCHEMA_VERSION,
        "call_id": call_id,
        "metadata": {"language": "en", "scenario": "ARTIFICIAL transcript-converter fixture"},
        "checks": [
            {
                "question_id": "FIX01",
                "expected": {"kind": "compliance", "expected_verdict": "FAIL"},
                "evidence_plan": {
                    "mode": "absence",
                    "scope": "entire_dialogue",
                    "required_speaker": "ANY",
                    "absent_event": "ARTIFICIAL absent marker",
                    "content_constraints": [],
                    "forbidden_patterns": [],
                },
                "gold_evidence_turns": [],
                "gold_evidence_text": "",
            }
        ],
        "dialogue": [
            {"turn_id": index, "speaker": speaker, "text": text, "critical_tokens": []}
            for index, (speaker, text) in enumerate(turns)
        ],
    }
