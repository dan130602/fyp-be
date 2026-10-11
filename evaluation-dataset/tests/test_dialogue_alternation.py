"""ARTIFICIAL dialogue-shape tests plus a guard over the real authored dialogues."""

import json
from pathlib import Path

import pytest

from compilation_fixtures import SCHEMAS, compilation_inputs
from fyp_evaluation.authoring.compile import compile_ground_truth
from fyp_evaluation.validation import validate_speaker_alternation

DIALOGUES = Path(__file__).resolve().parents[1] / "authoring" / "dialogues"


def _dialogue(*speakers):
    return {"turns": [{"speaker": speaker, "text": f"Turn {index}.", "critical_tokens": []}
                      for index, speaker in enumerate(speakers)]}


def test_alternating_dialogue_is_valid_from_either_speaker():
    assert validate_speaker_alternation(_dialogue("EMPLOYEE", "CLIENT", "EMPLOYEE", "CLIENT")) == []
    assert validate_speaker_alternation(_dialogue("CLIENT", "EMPLOYEE", "CLIENT")) == []
    assert validate_speaker_alternation(_dialogue("EMPLOYEE")) == []


@pytest.mark.parametrize(
    ("speakers", "expected"),
    [
        (("EMPLOYEE", "EMPLOYEE", "CLIENT"), ["turns 0-1: 2 consecutive EMPLOYEE turns"]),
        (("CLIENT", "EMPLOYEE", "EMPLOYEE", "EMPLOYEE"), ["turns 1-3: 3 consecutive EMPLOYEE turns"]),
        (("CLIENT", "CLIENT", "EMPLOYEE", "EMPLOYEE"),
         ["turns 0-1: 2 consecutive CLIENT turns", "turns 2-3: 2 consecutive EMPLOYEE turns"]),
    ],
)
def test_each_run_of_the_same_speaker_is_reported_once_by_zero_based_position(speakers, expected):
    errors = validate_speaker_alternation(_dialogue(*speakers))
    assert [error.split(";")[0] for error in errors] == expected
    assert all("merge them into one turn" in error for error in errors)


def test_compiler_rejects_consecutive_same_speaker_turns_and_writes_nothing():
    config, plan, dialogue, *rest = compilation_inputs()
    dialogue["turns"].insert(1, {**dialogue["turns"][0], "text": "ARTIFICIAL continuation."})
    with pytest.raises(ValueError, match="consecutive EMPLOYEE turns"):
        compile_ground_truth(config, plan, dialogue, *rest, schema_dir=SCHEMAS)


@pytest.mark.parametrize("path", sorted(DIALOGUES.glob("CALL_*.json")), ids=lambda path: path.stem)
def test_every_authored_pilot_dialogue_alternates_speakers(path):
    dialogue = json.loads(path.read_text(encoding="utf-8"))
    assert validate_speaker_alternation(dialogue) == []
