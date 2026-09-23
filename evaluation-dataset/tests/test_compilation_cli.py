"""CLI regression tests with ARTIFICIAL files in pytest temporary directories only."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from compilation_fixtures import SCHEMAS, compilation_inputs
from fyp_evaluation.validation.__main__ import main
from fyp_evaluation.validation.schemas import load_json, validate
from fyp_evaluation.authoring.coverage import compile_expected


@pytest.fixture
def files(tmp_path):
    config, plan, dialogue, checklist, normalization, matrix = compilation_inputs()
    paths = {}
    for name, data in (("call_config", config), ("check_plan", plan), ("dialogue", dialogue),
                       ("checklist", checklist), ("normalization", normalization)):
        # The deliberately misleading dialogue filename must not select its column.
        paths[name] = tmp_path / ("CALL_998.json" if name == "dialogue" else name + ".json")
        paths[name].write_text(json.dumps(data), encoding="utf-8")
    paths["matrix"] = tmp_path / "artificial_matrix.csv"
    with paths["matrix"].open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["question_id", *matrix])
        for item in checklist:
            writer.writerow([item["id"], *(cells[item["id"]] for cells in matrix.values())])
    paths["output"] = tmp_path / "artificial_output" / "ground_truth.json"
    return paths


def run(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["fyp-validate", "--schema-dir", str(SCHEMAS), *map(str, arguments)])
    return main()


def trailing(files):
    return [files["matrix"], files["checklist"], files["normalization"], "--call-ids", "CALL_999", "CALL_998"]


def compiler_args(files):
    return ["compile-ground-truth", files["call_config"], files["check_plan"], files["dialogue"], files["output"], *trailing(files)]


def test_standalone_dialogue_command_rejects_token_absent_from_own_turn(files):
    dialogue = load_json(files["dialogue"])
    dialogue["turns"][0]["critical_tokens"] = [dialogue["turns"][-1]["text"]]
    files["dialogue"].write_text(json.dumps(dialogue), encoding="utf-8")
    project = SCHEMAS.parent
    env = {**os.environ, "PYTHONPATH": str(project / "src")}
    result = subprocess.run([sys.executable, "-m", "fyp_evaluation.validation", "--schema-dir", str(SCHEMAS),
                             "json", "dialogue", str(files["dialogue"])],
                            capture_output=True, text=True, cwd=project, env=env, check=False)
    assert result.returncode == 1
    assert "Turn 0: critical token is not in spoken text" in result.stderr


def test_dialogue_plan_and_token_commands_and_misleading_filename(files, monkeypatch):
    assert run(monkeypatch, "dialogue-plan", files["dialogue"], files["check_plan"]) == 0
    assert run(monkeypatch, "critical-tokens", files["dialogue"], *trailing(files)) == 0
    dialogue = load_json(files["dialogue"])
    dialogue["call_id"] = "CALL_998"
    files["dialogue"].write_text(json.dumps(dialogue), encoding="utf-8")
    assert run(monkeypatch, "dialogue-plan", files["dialogue"], files["check_plan"]) == 1
    assert run(monkeypatch, "critical-tokens", files["dialogue"], *trailing(files)) == 1


def test_advisory_cli_success(files, monkeypatch, capsys):
    dialogue = load_json(files["dialogue"])
    dialogue["turns"].append({"speaker": "EMPLOYEE", "text": "ARTIFICIAL 123 and 5%", "critical_tokens": []})
    files["dialogue"].write_text(json.dumps(dialogue), encoding="utf-8")
    assert run(monkeypatch, "critical-tokens", files["dialogue"], *trailing(files)) == 0
    assert "Advisory (non-blocking)" in capsys.readouterr().out


def test_compiler_cli_validates_before_writing_and_requires_overwrite(files, monkeypatch):
    command = compiler_args(files)
    assert run(monkeypatch, *command) == 0
    output = files["output"]
    before = output.read_bytes()
    assert b"\r\n" not in before
    assert load_json(output)["call_id"] == "CALL_999"  # Not the dialogue filename.
    assert validate(load_json(output), "ground_truth", SCHEMAS) == []
    assert run(monkeypatch, *command) == 2
    assert output.read_bytes() == before
    assert run(monkeypatch, *command, "--overwrite") == 0
    assert output.read_bytes() == before
    dialogue = load_json(files["dialogue"])
    dialogue["call_id"] = "CALL_998"
    files["dialogue"].write_text(json.dumps(dialogue), encoding="utf-8")
    assert run(monkeypatch, *command, "--overwrite") == 2
    assert output.read_bytes() == before


def test_invalid_compilation_creates_no_output(files, monkeypatch):
    plan = load_json(files["check_plan"])
    plan["checks"][0]["evidence_plan"]["planned_evidence"][0]["text"] = "ARTIFICIAL unmatched"
    files["check_plan"].write_text(json.dumps(plan), encoding="utf-8")
    assert run(monkeypatch, *compiler_args(files)) == 2
    assert not files["output"].parent.exists()


def test_output_cannot_replace_authoring_input(files, monkeypatch):
    files["output"] = files["dialogue"]
    before = files["dialogue"].read_bytes()
    assert run(monkeypatch, *compiler_args(files), "--overwrite") == 2
    assert files["dialogue"].read_bytes() == before


@pytest.mark.parametrize("qid,phrase,valid", [
    ("FIX_DOB", "the fourth of August, nineteen ninety-one", True),
    ("FIX_DOB", "the fifth of August, nineteen ninety-one", False),
    ("K01", "S zero zero zero zero zero zero zero X", True),
    ("K01", "S zero zero zero zero zero zero one X", False),
    ("FIX_CURRENCY", "E, U, R", True),
    ("FIX_CURRENCY", "G, B, P", False),
    ("FIX_INSTRUMENT", "The ARTIFICIAL Copper Basket", True),
    ("FIX_INSTRUMENT", "The ARTIFICIAL Silver Basket", False),
    # Regression: a longer identifier containing the locked value as a
    # substring, and a compound day-word containing the locked day's own
    # word as a suffix, must both be rejected by the actual CLI subprocess.
    ("K01", "XS0000000X", False),
    ("FIX_DOB", "the twenty fourth of August, nineteen ninety-one", False),
])
def test_actual_check_plan_complete_cli_verifies_spoken_values(files, qid, phrase, valid):
    # Explicit user DOB example in ARTIFICIAL temporary inputs only. Expected
    # objects are compiled from the artificial CSV cell, not from the phrase.
    with files["matrix"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    for row in rows[1:]:
        if row[0] == "FIX_DOB":
            row[1:] = ["04 August 1991"] * (len(row) - 1)
    with files["matrix"].open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    checklist, normalization = load_json(files["checklist"]), load_json(files["normalization"])
    dob_item = next(item for item in checklist if item["id"] == "FIX_DOB")
    for name, key in (("call_config", "intended_answers"), ("check_plan", "checks")):
        document = load_json(files[name])
        dob = next(answer for answer in document[key] if answer["question_id"] == "FIX_DOB")
        dob["expected"] = compile_expected(dob_item, normalization, "04 August 1991")
        if name == "check_plan":
            dob["evidence_plan"]["planned_evidence"][0]["text"] = "4 August 1991"
            target = next(check for check in document["checks"] if check["question_id"] == qid)
            target["evidence_plan"]["planned_evidence"][0]["text"] = phrase
        files[name].write_text(json.dumps(document), encoding="utf-8")
    before = {name: path.read_bytes() for name, path in files.items() if name != "output"}
    command = [sys.executable, "-m", "fyp_evaluation.validation", "--schema-dir", str(SCHEMAS),
               "check-plan", str(files["check_plan"]), str(files["call_config"]),
               *map(str, trailing(files))]
    kwargs = {"capture_output": True, "text": True, "cwd": SCHEMAS.parent,
              "env": {**os.environ, "PYTHONPATH": str(SCHEMAS.parent / "src")}, "check": False}
    # Scaffold mode retains its earlier structural/value checks.
    assert subprocess.run(command, **kwargs).returncode == 0
    result = subprocess.run([*command, "--complete"], **kwargs)
    assert result.returncode == (0 if valid else 1), result.stderr
    if not valid:
        assert f"{qid}: planned evidence does not denote locked extraction value" in result.stderr
    assert {name: files[name].read_bytes() for name in before} == before
    assert not files["output"].exists()
