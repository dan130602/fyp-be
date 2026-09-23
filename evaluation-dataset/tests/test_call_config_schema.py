"""Artificial schema fixtures for the Stage 3A contract amendment."""

from pathlib import Path

import pytest

from fixtures import documents
from fyp_evaluation.validation import validate

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"


@pytest.mark.parametrize("value", [None, "", " \t\n", 1, False])
def test_scenario_intent_requires_nonblank_string(value):
    data = documents()["call_config"]
    data["intended_answers"][0]["scenario_intent"] = value
    assert validate(data, "call_config", SCHEMAS)


def test_scenario_intent_is_required_but_placeholder_is_structurally_valid():
    data = documents()["call_config"]
    item = data["intended_answers"][0]
    item["scenario_intent"] = "TODO_AUTHOR_SCENARIO_INTENT"
    assert validate(data, "call_config", SCHEMAS) == []
    del item["scenario_intent"]
    assert validate(data, "call_config", SCHEMAS)


@pytest.mark.parametrize("name", ["call_config", "check_plan", "ground_truth"])
def test_stage_3a_shared_version_requires_explicit_migration(name):
    data = documents()[name]
    assert data["schema_version"] == ("0.5.0" if name == "call_config" else "0.6.0")
    data["schema_version"] = "0.4.0"
    assert validate(data, name, SCHEMAS)


@pytest.mark.parametrize("field", ["evidence_plan", "turn_id", "timestamp", "dialogue"])
def test_call_intent_does_not_admit_evidence_or_dialogue_fields(field):
    data = documents()["call_config"]
    data["intended_answers"][0][field] = "ARTIFICIAL"
    assert validate(data, "call_config", SCHEMAS)
