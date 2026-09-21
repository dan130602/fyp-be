"""ARTIFICIAL SCHEMA FIXTURES ONLY. Not checklist or evaluation dataset content."""

from copy import deepcopy


def documents():
    base = {"schema_version": "0.1.0", "call_id": "CALL_999"}
    expected = {"kind": "compliance", "expected_verdict": "PASS"}
    check = {
        "question_id": "FIX_MARKER", "expected": expected,
        "evidence_plan": {
            "mode": "presence",
            "planned_evidence": [{
                "evidence_id": "FIX_PHRASE", "required_speaker": "EMPLOYEE",
                "text": "artificial marker",
            }],
            "content_constraints": [],
        },
    }
    turn = {"speaker": "EMPLOYEE", "text": "artificial marker", "critical_tokens": ["marker"]}
    metadata = {"language": "en", "scenario": "ARTIFICIAL SCHEMA FIXTURE ONLY"}
    return deepcopy({
        "checklist": [{
            "id": "FIX_MARKER", "type": "COMPLIANCE", "requirement_type": "OBLIGATION",
            "text": "Was the artificial marker stated?",
        }],
        "call_config": {
            **base, "schema_version": "0.4.0", "metadata": metadata,
            "target_duration_seconds": {"min": 1, "max": 2},
            "scenario_facts": [],
            "intended_answers": [{"question_id": "FIX_MARKER", "expected": expected}],
        },
        "check_plan": {**base, "schema_version": "0.4.0", "checks": [check]},
        "dialogue": {**base, "turns": [turn]},
        "ground_truth": {
            **base, "schema_version": "0.4.0", "metadata": metadata,
            "checks": [{**check, "gold_evidence_turns": [0], "gold_evidence_text": "artificial marker"}],
            "dialogue": [{"turn_id": 0, **turn}],
        },
        "timings": {
            **base, "render_id": "fixture_render", "audio_file": "audio.mp3",
            "duration_seconds": 2,
            "turns": [{"turn_id": 0, "start_seconds": 0, "end_seconds": 1,
                       "words": [{"text": "artificial", "start_seconds": 0, "end_seconds": 0.5}]}],
        },
        "normalization": {
            "schema_version": "0.3.0",
            "rules": {"fixture_exact": {"value_type": "verdict", "operation": "exact"}},
            "compliance_rule": "fixture_exact", "field_rules": {},
        },
    })


def mixed_documents():
    """Artificial mixed-category fixtures, never real gold or generated calls."""
    data = documents()
    values = [
        ("FIX_DOB", "CLIENT_KYC", "dob", "dob_dd_month_yyyy", "01 January 2000"),
        ("FIX_QUANTITY", "TRANSACTION", "fixture_quantity", "integer", 7),
        ("FIX_LABEL", "TRANSACTION", "fixture_label", "text", "artificial label"),
    ]
    for question_id, category, field, value_type, value in values:
        data["checklist"].append({
            "id": question_id, "type": category, "field": field,
            "text": f"ARTIFICIAL fixture extraction: {field}?",
        })
        rule_id = f"fixture_{value_type}"
        data["normalization"]["rules"][rule_id] = {"value_type": value_type, "operation": "exact"}
        data["normalization"]["field_rules"][field] = rule_id
        expected = {"kind": "extraction", "value_type": value_type, "expected_value": value}
        for name in ["call_config", "check_plan", "ground_truth"]:
            key = "intended_answers" if name == "call_config" else "checks"
            item = deepcopy(data[name][key][0])
            item.update(question_id=question_id, expected=deepcopy(expected))
            data[name][key].append(item)
    return data
