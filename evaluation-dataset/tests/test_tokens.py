"""ARTIFICIAL Entity WER annotation fixtures; no scorer or ASR alignment."""

from compilation_fixtures import SCHEMAS, compilation_inputs
from fyp_evaluation.authoring.compile import compile_ground_truth
from fyp_evaluation.validation.evidence import validate_dialogue_against_plan
from fyp_evaluation.validation.tokens import validate_critical_token_coverage


def coverage(inputs):
    return validate_critical_token_coverage(inputs[2], *inputs[3:], schema_dir=SCHEMAS)


def test_annotations_outside_evidence_and_evidence_without_tokens_are_valid():
    inputs = compilation_inputs()
    for turn in inputs[2]["turns"]:
        turn["critical_tokens"] = []
    outside = "ARTIFICIAL annotations: S0000000X; 1 May 1982; ARTIFICIAL Copper Basket; EUR; fee 2%."
    inputs[2]["turns"].append({"speaker": "EMPLOYEE", "text": outside, "critical_tokens": [outside]})
    assert validate_dialogue_against_plan(inputs[2], inputs[1]) == []
    errors, advisory = coverage(inputs)
    assert errors == []
    assert advisory  # Unannotated occurrences elsewhere remain a review aid only.
    ground = compile_ground_truth(*inputs, schema_dir=SCHEMAS)
    assert ground["dialogue"][-1]["critical_tokens"] == [outside]
    assert all(not turn["critical_tokens"] for turn in ground["dialogue"][:-1])


def test_no_token_count_cap():
    inputs = compilation_inputs()
    tokens = [f"ARTIFICIALterm{i}" for i in range(300)]
    inputs[2]["turns"].append({"speaker": "CLIENT", "text": " ".join(tokens), "critical_tokens": tokens})
    assert not coverage(inputs)[0]


def test_missing_non_null_extraction_coverage_fails():
    inputs = compilation_inputs()
    inputs[2]["turns"][-1]["critical_tokens"] = []
    errors, advisory = coverage(inputs)
    assert errors == ["FIX_CURRENCY: no critical token covers the locked extraction value"]
    assert advisory


def test_null_identifier_needs_no_token():
    assert not coverage(compilation_inputs(null_identifier=True))[0]


def test_advisory_report_is_nonblocking_and_has_no_evidence_role():
    inputs = compilation_inputs()
    inputs[2]["turns"].append({"speaker": "EMPLOYEE", "text": "ARTIFICIAL fee 12.5%, $1,234.50 and GBP 20.", "critical_tokens": []})
    errors, advisory = coverage(inputs)
    assert errors == []
    assert {record["text"] for record in advisory} >= {"12.5%", "$1,234.50", "GBP", "20"}
    assert compile_ground_truth(*inputs, schema_dir=SCHEMAS)


def test_tokens_reuse_schema_uniqueness_and_own_turn_rules():
    inputs = compilation_inputs()
    turn = inputs[2]["turns"][-1]
    turn["critical_tokens"] *= 2
    assert "non-unique" in " ".join(coverage(inputs)[0])
    turn["critical_tokens"] = ["ARTIFICIAL affirmative marker."]  # Exists only in another turn.
    assert coverage(inputs)[0] == [f"Turn {len(inputs[2]['turns']) - 1}: critical token is not in spoken text"]


def test_matrix_selection_from_validated_internal_id():
    inputs = compilation_inputs()
    inputs[2]["call_id"] = "CALL_998"  # Own column requires GBP, not EUR.
    assert "FIX_CURRENCY" in " ".join(coverage(inputs)[0])
    inputs[2]["turns"][-1].update(text="ARTIFICIAL GBP", critical_tokens=["GBP"])
    assert not coverage(inputs)[0]
    inputs[2]["call_id"] = "CALL_997"
    assert coverage(inputs)[0] == ["CALL_997: no column in approved matrix"]
