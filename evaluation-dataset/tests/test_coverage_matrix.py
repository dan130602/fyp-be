"""Checks of the fixed five-call pilot CSV, not a general matrix validator.

Reads the real predetermined answers without generating call artifacts.
Semantic feasibility still needs review and later independent verification.
"""

import csv
import re
from pathlib import Path

from fyp_evaluation.validation import load_json, validate_checklist
from fyp_evaluation.validation.dates import is_canonical_dob
from fyp_evaluation.authoring.coverage import parse_k01_cell

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets/legacy_pilot_v1"


def matrix_rows():
    with (DATASET_ROOT / "authoring/coverage_matrix.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.reader(file, strict=True))


def test_pilot_csv_structure_and_complete_checklist_coverage():
    rows = matrix_rows()
    checklist = load_json(DATASET_ROOT / "checklist.json")
    normalization = load_json(ROOT / "configs/normalization.json")
    assert validate_checklist(checklist, normalization, ROOT / "schemas") == []
    assert rows[0] == ["question_id", "CALL_001", "CALL_002", "CALL_003", "CALL_004", "CALL_005"]
    assert len(rows) == 11
    assert all(len(row) == 6 for row in rows)
    assert [row[0] for row in rows[1:]] == [item["id"] for item in checklist]
    assert len({row[0] for row in rows[1:]}) == 10
    assert all(cell and cell == cell.strip() for row in rows for cell in row)
    assert all("\n" not in cell and "\r" not in cell for row in rows for cell in row)


def test_pilot_verdict_balance_and_actual_extraction_values():
    matrix = {row[0]: row[1:] for row in matrix_rows()[1:]}
    for question_id in ["C01", "C02", "C03", "C04", "C05", "C06"]:
        values = matrix[question_id]
        assert set(values) == {"PASS", "FAIL"}
        assert values.count("PASS") >= 2
        assert values.count("FAIL") >= 2
    for question_id in ["K01", "K02", "T01", "T02"]:
        assert not set(matrix[question_id]) & {"PASS", "FAIL", "N/A", "UNCLEAR"}


def test_pilot_identifiers_are_explicitly_invalid_test_surrogates():
    matrix = {row[0]: row[1:] for row in matrix_rows()[1:]}
    assert matrix["K01"] == ["S9100001X", "S8400002X", "F7800003A", "G8800004A", "NOT_STATED"]
    identifiers = matrix["K01"][:4]
    assert len(set(identifiers)) == 4
    assert all(re.fullmatch(r"[STFGM][0-9]{7}[A-Z]", value) for value in identifiers)
    assert [parse_k01_cell(value) for value in matrix["K01"]] == [*identifiers, None]
    # Independent invalidity check: supplied suffixes cannot occur in the
    # published checksum alphabets, regardless of the seven digits.
    for value in identifiers:
        alphabet = "JZIHGFEDCBA" if value[0] in "ST" else "XWUTRQPNMLK"
        assert value[-1] not in alphabet


def test_other_nine_matrix_rows_remain_frozen():
    rows = [row for row in matrix_rows()[1:] if row[0] != "K01"]
    assert rows == [
        ["C01", "PASS", "PASS", "FAIL", "FAIL", "PASS"],
        ["C02", "PASS", "FAIL", "PASS", "FAIL", "PASS"],
        ["C03", "PASS", "FAIL", "FAIL", "FAIL", "PASS"],
        ["C04", "PASS", "PASS", "PASS", "FAIL", "FAIL"],
        ["C05", "PASS", "FAIL", "FAIL", "PASS", "PASS"],
        ["C06", "PASS", "PASS", "FAIL", "PASS", "FAIL"],
        ["K02", "04 August 1991", "13 May 1984", "22 November 1978", "09 February 1988", "17 September 1995"],
        ["T01", "TEST SGD Equity Fund", "TEST USD Equity Fund", "TEST EUR Equity Fund", "TEST GBP Equity Fund", "TEST AUD Equity Fund"],
        ["T02", "SGD", "USD", "EUR", "GBP", "AUD"],
    ]


def test_pilot_dobs_are_distinct_canonical_calendar_dates():
    matrix = {row[0]: row[1:] for row in matrix_rows()[1:]}
    assert len(set(matrix["K02"])) == 5
    assert all(is_canonical_dob(value) for value in matrix["K02"])


def test_pilot_fictional_funds_match_transaction_currencies():
    matrix = {row[0]: row[1:] for row in matrix_rows()[1:]}
    assert matrix["T02"] == ["SGD", "USD", "EUR", "GBP", "AUD"]
    assert matrix["T01"] == [f"TEST {currency} Equity Fund" for currency in matrix["T02"]]


def test_pilot_avoids_selected_ambiguous_combinations():
    matrix = {row[0]: row[1:] for row in matrix_rows()[1:]}
    # These constrain this reviewed matrix only; they are not general
    # applicability dependencies or a semantic verification algorithm.
    for index in range(5):
        if matrix["C06"][index] == "FAIL":
            assert matrix["C02"][index] == "PASS"
        if matrix["C05"][index] == "FAIL":
            assert matrix["C03"][index] == "FAIL"
