"""G2A call specs: pure deterministic compilation of the frozen G1 outputs.

Read-only tests use the committed release; tamper and write tests work on
temporary copies. No LLM, network or AWS service is used.
"""

import csv
import io
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path

import pytest

from fyp_evaluation.call_specs import compile as compile_module
from fyp_evaluation.call_specs.cli import main
from fyp_evaluation.call_specs.compile import (
    G1StateError, RECORD, SPEC_DIR, load_g1_sources, parse_json_text, project_to_g1_texts, read_sources, serialize_json,
    spec_path, text_hash,
)
from fyp_evaluation.call_specs.outputs import check_outputs, generate, read_disk, write_specs
from fyp_evaluation.call_specs.validate import check_call_spec_schemas, validate_spec_set
from fyp_evaluation.planning import outputs as g1

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
RELEASE = ROOT / "datasets" / "advisory_execution_pilot_v1"
CALL_IDS = [f"CALL_{n:03d}" for n in range(1, 51)]
PYTHON_TYPES = {"text": str, "dob_dd_month_yyyy": str, "integer": int, "number": int}


@pytest.fixture(scope="module")
def src():
    return read_sources(RELEASE)


@pytest.fixture(scope="module")
def disk_specs():
    return {cid: parse_json_text((RELEASE / spec_path(cid)).read_text(encoding="utf-8")) for cid in CALL_IDS}


def _copy_release(source, destination):
    """Copy file contents into plain new directories.

    shutil.copytree also copies directory attributes; on a OneDrive-synced Windows tree those include
    ReadOnly, which then makes later rmtree calls fail. Contents are all the tests need.
    """
    for path in sorted(source.rglob("*")):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def _remove_tree(path):
    """rmtree that also clears a Windows ReadOnly attribute instead of failing on it."""
    def clear(function, target, *_):
        os.chmod(target, stat.S_IWRITE)
        function(target)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=clear)
    else:
        shutil.rmtree(path, onerror=clear)


def _symlink(link, target):
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks are unavailable in this environment: {error}")


def _snapshot(release):
    """Bytes of every generated file, to prove a refused run wrote nothing."""
    files = {p.name: p.read_bytes() for p in (release / SPEC_DIR).iterdir() if p.is_file() and not p.is_symlink()}
    files[RECORD] = (release / RECORD).read_bytes()
    return files


@pytest.fixture()
def release(tmp_path):
    """A writable copy of the release including the generated G2A outputs."""
    destination = tmp_path / "advisory_execution_pilot_v1"
    destination.mkdir()
    _copy_release(RELEASE, destination)
    return destination


def _edit_spec(release, call_id, change):
    path = release / spec_path(call_id)
    spec = parse_json_text(path.read_text(encoding="utf-8"))
    change(spec)
    path.write_text(serialize_json(spec), encoding="utf-8", newline="\n")


def _validate(release):
    return validate_spec_set(read_sources(release), read_disk(release), SCHEMAS)


def _first_call_with_edge(disk_specs):
    return next(cid for cid in CALL_IDS if disk_specs[cid]["edge_cases"])


def test_schemas_are_valid():
    check_call_spec_schemas(SCHEMAS)


def test_exactly_50_call_specs_exist():
    files = sorted(p.name for p in (RELEASE / SPEC_DIR).iterdir())
    assert files == [f"{cid}.json" for cid in CALL_IDS]
    assert (RELEASE / RECORD).is_file()


def test_one_spec_per_manifest_call(src, disk_specs):
    assert src.call_ids == CALL_IDS
    assert [disk_specs[cid]["call_id"] for cid in CALL_IDS] == CALL_IDS


def test_twenty_gold_entries_in_checklist_order(src, disk_specs):
    checklist_ids = [item["id"] for item in src.documents["checklist"]]
    assert len(checklist_ids) == 20
    for spec in disk_specs.values():
        assert [g["question_id"] for g in spec["gold"]] == checklist_ids


def test_json_types_are_preserved(disk_specs):
    for spec in disk_specs.values():
        for entry in spec["gold"]:
            expected = entry["expected"]
            if expected["kind"] == "compliance":
                assert expected["expected_verdict"] in {"PASS", "FAIL"}
            else:
                assert type(expected["expected_value"]) is PYTHON_TYPES[expected["value_type"]], entry
    value_types = {e["question_id"]: e["expected"].get("value_type") for e in disk_specs["CALL_001"]["gold"]}
    assert value_types == {
        **{f"C{n:02d}": None for n in range(1, 13)},
        "K01": "text", "K02": "dob_dd_month_yyyy", "T01": "text", "T02": "text",
        "T03": "number", "T04": "integer", "T05": "text", "T06": "text",
    }


def test_t03_never_becomes_a_float():
    for cid in CALL_IDS:
        text = (RELEASE / spec_path(cid)).read_text(encoding="utf-8")
        assert not re.search(r'"expected_value": -?\d+\.\d', text), cid
        spec = parse_json_text(text)
        t03 = next(g for g in spec["gold"] if g["question_id"] == "T03")["expected"]
        assert t03["value_type"] == "number" and type(t03["expected_value"]) is int


def test_k01_is_never_null(disk_specs):
    for spec in disk_specs.values():
        k01 = next(g for g in spec["gold"] if g["question_id"] == "K01")["expected"]
        assert isinstance(k01["expected_value"], str) and k01["expected_value"]


def test_call_intents_is_an_array_in_vocabulary_order(src, disk_specs):
    vocabulary = src.documents["generation_plan"]["call_intent_vocabulary"]
    for spec in disk_specs.values():
        intents = spec["metadata"]["call_intents"]
        assert isinstance(intents, list) and intents == ["ADVISORY", "EXECUTION"]
        assert intents == sorted(intents, key=vocabulary.index)


def test_no_obsolete_singular_intent_field(disk_specs):
    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)
    for spec in disk_specs.values():
        assert not {"gold_call_intent", "call_intent"} & set(keys(spec))


def test_every_assignment_occurs_exactly_once_in_its_call(src, disk_specs):
    placed = [(cid, e["assignment_id"]) for cid, spec in disk_specs.items() for e in spec["edge_cases"]]
    assert sorted(placed) == sorted((a["call_id"], a["assignment_id"]) for a in src.assignments)
    assert len(placed) == len({assignment for _, assignment in placed}) == 90


def test_edge_case_parameters_are_unchanged(disk_specs):
    rows = list(csv.DictReader(io.StringIO((RELEASE / g1.ASSIGNMENTS).read_text(encoding="utf-8"))))
    by_id = {row["assignment_id"]: row for row in rows}
    for spec in disk_specs.values():
        for edge in spec["edge_cases"]:
            row = by_id[edge["assignment_id"]]
            assert g1.canonical_json(edge["parameters"]) == row["parameters_json"]
            assert (edge["edge_case_id"], edge["family"], "|".join(edge["target_question_ids"])) == (
                row["edge_case_id"], row["family"], row["target_question_ids"])


def test_specs_project_back_to_all_three_g1_csvs_byte_for_byte(src, disk_specs):
    projected = project_to_g1_texts([disk_specs[cid] for cid in CALL_IDS], src.documents)
    planning = json.loads((RELEASE / g1.RECORD).read_text(encoding="utf-8"))
    assert set(projected) == {g1.MANIFEST, g1.MATRIX, g1.ASSIGNMENTS}
    for rel, text in projected.items():
        frozen = (RELEASE / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")
        assert text == frozen, rel
        assert g1.sha256(text.encode("utf-8")) == planning["outputs"][rel], rel


def test_source_hashes_match_the_frozen_g1_files(src, disk_specs):
    for spec in disk_specs.values():
        assert spec["source"] == src.hashes
    assert src.hashes["planning_record_sha256"] == g1.sha256((RELEASE / g1.RECORD).read_bytes())


def test_record_summarises_the_set(disk_specs):
    record = json.loads((RELEASE / RECORD).read_text(encoding="utf-8"))
    assert record["stage"] == "G2A" and list(record["call_specs"]) == CALL_IDS
    assert record["summary"] == {
        "call_specs": 50, "gold_entries_per_call": 20, "edge_case_assignments": 90,
        "calls_with_edge_cases": 40, "calls_with_compliance_fails": 40,
    }
    for cid in CALL_IDS:
        assert record["call_specs"][cid] == g1.sha256((RELEASE / spec_path(cid)).read_bytes())
    assert "timestamp" not in json.dumps(record).lower()


def test_committed_outputs_validate_and_reproduce():
    assert main(["--dataset-root", str(RELEASE), "--validate"]) == 0
    assert main(["--dataset-root", str(RELEASE), "--check"]) == 0


def test_generation_is_deterministic():
    first = generate(RELEASE, SCHEMAS)[1]
    second = generate(RELEASE, SCHEMAS)[1]
    assert first == second and len(first) == 51
    assert check_outputs(RELEASE, first) == {"missing": [], "changed": [], "extra": []}


def test_second_cli_generation_is_byte_identical(release, capsys):
    before = {p.name: p.read_bytes() for p in (release / SPEC_DIR).iterdir()} | {RECORD: (release / RECORD).read_bytes()}
    assert main(["--dataset-root", str(release), "--overwrite"]) == 0
    after = {p.name: p.read_bytes() for p in (release / SPEC_DIR).iterdir()} | {RECORD: (release / RECORD).read_bytes()}
    assert before == after


def test_existing_outputs_are_not_overwritten_without_flag(release, capsys):
    before = (release / spec_path("CALL_001")).read_bytes()
    assert main(["--dataset-root", str(release)]) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert (release / spec_path("CALL_001")).read_bytes() == before


def test_fresh_release_generates_once_then_refuses(release):
    _remove_tree(release / SPEC_DIR)
    (release / RECORD).unlink()
    assert main(["--dataset-root", str(release), "--validate"]) == 1
    assert main(["--dataset-root", str(release)]) == 0
    assert main(["--dataset-root", str(release), "--validate"]) == 0
    assert main(["--dataset-root", str(release)]) == 2


def test_stale_g1_manifest_is_refused(release, capsys):
    path = release / g1.MANIFEST
    path.write_bytes(path.read_bytes().replace(b"en-SG", b"en-GB", 1))
    assert main(["--dataset-root", str(release), "--overwrite"]) == 2
    assert "G1 outputs are not valid" in capsys.readouterr().err
    with pytest.raises(G1StateError):
        load_g1_sources(release, SCHEMAS)
    assert main(["--dataset-root", str(release), "--validate"]) == 2


def test_stale_g1_after_g0_change_is_refused(release):
    path = release / "authoring" / "generation_plan.json"
    path.write_bytes(path.read_bytes().replace(b"G1 planning distribution", b"G1 planning distributions", 1))
    with pytest.raises(G1StateError, match="G1 outputs are not valid"):
        load_g1_sources(release, SCHEMAS)
    assert main(["--dataset-root", str(release), "--overwrite"]) == 2


def test_mutated_gold_value_fails_validation(release):
    def change(spec):
        entry = next(g for g in spec["gold"] if g["question_id"] == "T04")
        entry["expected"]["expected_value"] += 1
    _edit_spec(release, "CALL_001", change)
    errors = _validate(release)
    assert any("gold T04 differs from coverage_matrix.csv" in e for e in errors), errors
    assert main(["--dataset-root", str(release), "--validate"]) == 1


def test_mutated_verdict_fails_validation(release):
    def change(spec):
        entry = next(g for g in spec["gold"] if g["question_id"] == "C01")
        entry["expected"]["expected_verdict"] = "FAIL" if entry["expected"]["expected_verdict"] == "PASS" else "PASS"
    _edit_spec(release, "CALL_002", change)
    assert any("gold C01 differs" in e for e in _validate(release))


def test_float_amount_fails_even_though_numerically_equal(release):
    def change(spec):
        entry = next(g for g in spec["gold"] if g["question_id"] == "T03")
        entry["expected"]["expected_value"] = float(entry["expected"]["expected_value"])
    _edit_spec(release, "CALL_003", change)
    errors = _validate(release)
    assert any("floating-point number" in e for e in errors) and any("gold T03 differs" in e for e in errors), errors


def test_null_k01_is_rejected(release):
    def change(spec):
        next(g for g in spec["gold"] if g["question_id"] == "K01")["expected"]["expected_value"] = None
    _edit_spec(release, "CALL_004", change)
    assert any("CALL_004" in e and "None" in e for e in _validate(release))


def test_dropped_edge_assignment_fails(release, disk_specs):
    call_id = _first_call_with_edge(disk_specs)
    _edit_spec(release, call_id, lambda spec: spec["edge_cases"].pop())
    errors = _validate(release)
    assert any("edge-case" in e or "edge cases" in e for e in errors), errors


def test_edited_edge_parameter_fails(release, disk_specs):
    call_id = _first_call_with_edge(disk_specs)
    _edit_spec(release, call_id, lambda spec: spec["edge_cases"][0]["parameters"].update(tampered=1))
    assert any("differs from edge_case_assignments.csv" in e for e in _validate(release))


def test_edge_case_moved_to_another_call_fails(release, disk_specs):
    donor = _first_call_with_edge(disk_specs)
    taker = next(cid for cid in CALL_IDS if cid != donor and not disk_specs[cid]["edge_cases"])
    moved = {}
    _edit_spec(release, donor, lambda spec: moved.update(edge=spec["edge_cases"].pop(0)))
    _edit_spec(release, taker, lambda spec: spec["edge_cases"].append(moved["edge"]))
    assert _validate(release)


def test_stale_source_hash_fails(release):
    _edit_spec(release, "CALL_005", lambda spec: spec["source"].update(checklist_sha256="0" * 64))
    assert any("source hashes differ" in e for e in _validate(release))


def test_unexpected_extra_file_fails_validate_and_check(release):
    (release / SPEC_DIR / "CALL_051.json").write_text("{}", encoding="utf-8")
    assert any("unexpected file" in e for e in _validate(release))
    assert main(["--dataset-root", str(release), "--validate"]) == 1
    assert main(["--dataset-root", str(release), "--check"]) == 1


def test_missing_file_and_missing_record_fail(release):
    (release / spec_path("CALL_007")).unlink()
    (release / RECORD).unlink()
    errors = _validate(release)
    assert any("missing authoring/call_specs/CALL_007.json" in e for e in errors)
    assert any(f"missing {RECORD}" in e for e in errors)
    assert main(["--dataset-root", str(release), "--check"]) == 1


def test_tampered_record_fails(release):
    path = release / RECORD
    record = json.loads(path.read_text(encoding="utf-8"))
    record["call_specs"]["CALL_010"] = "f" * 64
    path.write_text(serialize_json(record), encoding="utf-8", newline="\n")
    assert any("call_specs differs from the value recomputed" in e for e in _validate(release))


def test_hand_edited_formatting_is_rejected(release):
    path = release / spec_path("CALL_011")
    path.write_text(path.read_text(encoding="utf-8").replace("\n  ", "\n    ", 1), encoding="utf-8", newline="\n")
    assert any("not the canonical serialization" in e for e in _validate(release))


def test_overwrite_removes_stale_extra_files(release):
    stale = release / SPEC_DIR / "CALL_999.json"
    stale.write_text("{}", encoding="utf-8")
    (release / SPEC_DIR / "CALL_001.json.tmp").write_text("partial", encoding="utf-8")
    assert main(["--dataset-root", str(release), "--overwrite"]) == 0
    assert sorted(p.name for p in (release / SPEC_DIR).iterdir()) == [f"{cid}.json" for cid in CALL_IDS]
    assert main(["--dataset-root", str(release), "--validate"]) == 0


def test_write_refuses_unexpected_subdirectory(release):
    (release / SPEC_DIR / "nested").mkdir()
    texts = generate(release, SCHEMAS)[1]
    with pytest.raises(ValueError, match="subdirectory"):
        write_specs(release, texts, overwrite=True)


def test_g1_projection_does_not_trust_the_record(release):
    """Changing a gold cell and fixing every hash inside the specs still fails against the frozen G1 CSV."""
    def change(spec):
        entry = next(g for g in spec["gold"] if g["question_id"] == "T06")
        entry["expected"]["expected_value"] = "LIMIT" if entry["expected"]["expected_value"] == "MARKET" else "MARKET"
    _edit_spec(release, "CALL_020", change)
    assert any("gold T06 differs" in e for e in _validate(release))


def test_record_temp_symlink_is_refused_and_its_target_untouched(release, tmp_path, capsys):
    victim = tmp_path / "victim.txt"
    victim.write_text("PRECIOUS CONTENT", encoding="utf-8")
    link = release / (RECORD + ".tmp")
    _symlink(link, victim)
    before = _snapshot(release)
    for arguments in ([], ["--overwrite"]):
        assert main(["--dataset-root", str(release), *arguments]) == 2
        assert "symlink" in capsys.readouterr().err
    assert victim.read_text(encoding="utf-8") == "PRECIOUS CONTENT"
    assert link.is_symlink()
    assert _snapshot(release) == before


def test_spec_directory_temp_symlink_is_refused_and_its_target_untouched(release, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("PRECIOUS CONTENT", encoding="utf-8")
    link = release / SPEC_DIR / "CALL_001.json.tmp"
    _symlink(link, victim)
    texts = generate(release, SCHEMAS)[1]
    with pytest.raises(ValueError, match="symlink"):
        write_specs(release, texts, overwrite=True)
    assert victim.read_text(encoding="utf-8") == "PRECIOUS CONTENT" and link.is_symlink()


def test_expected_output_path_symlink_is_rejected(release, tmp_path):
    spec = release / spec_path("CALL_003")
    target = tmp_path / "elsewhere.json"
    shutil.copyfile(spec, target)
    spec.unlink()
    _symlink(spec, target)
    original = target.read_bytes()
    assert any("CALL_003" in e and "not a regular file" in e for e in _validate(release))
    assert main(["--dataset-root", str(release), "--validate"]) == 1
    assert main(["--dataset-root", str(release), "--overwrite"]) == 2
    assert target.read_bytes() == original and spec.is_symlink()


def test_orphaned_record_temp_is_detected_then_replaced(release, capsys):
    orphan = release / (RECORD + ".tmp")
    orphan.write_text("left behind by an interrupted run", encoding="utf-8")
    errors = _validate(release)
    assert any("orphaned temporary file" in e and RECORD in e for e in errors), errors
    assert main(["--dataset-root", str(release), "--validate"]) == 1
    assert main(["--dataset-root", str(release), "--check"]) == 1
    assert "extra: " + RECORD + ".tmp" in capsys.readouterr().err
    assert main(["--dataset-root", str(release), "--overwrite"]) == 0
    assert not orphan.exists()
    assert main(["--dataset-root", str(release), "--validate"]) == 0
    assert main(["--dataset-root", str(release), "--check"]) == 0


def test_orphaned_record_temp_blocks_generation_without_overwrite(release):
    _remove_tree(release / SPEC_DIR)
    (release / RECORD).unlink()
    orphan = release / (RECORD + ".tmp")
    orphan.write_text("interrupted", encoding="utf-8")
    assert main(["--dataset-root", str(release)]) == 2
    assert orphan.read_text(encoding="utf-8") == "interrupted" and not (release / SPEC_DIR).exists()


@pytest.mark.parametrize("reorder", ["top_level", "metadata", "gold_entry"])
def test_reordered_json_with_a_matching_record_is_rejected_by_validate(release, reorder, capsys):
    """The record is not authority: equivalent JSON in non-canonical byte form must still fail --validate."""
    path = release / spec_path("CALL_001")
    original = parse_json_text(path.read_text(encoding="utf-8"))
    spec = parse_json_text(path.read_text(encoding="utf-8"))
    if reorder == "top_level":
        spec = {key: spec[key] for key in sorted(spec)}
    elif reorder == "metadata":
        spec["metadata"] = dict(reversed(list(spec["metadata"].items())))
    else:
        first = spec["gold"][0]
        spec["gold"][0] = {"expected": first["expected"], "question_id": first["question_id"]}
    path.write_text(serialize_json(spec), encoding="utf-8", newline="\n")
    assert parse_json_text(path.read_text(encoding="utf-8")) == original  # semantically identical
    assert path.read_bytes() != (RELEASE / spec_path("CALL_001")).read_bytes()  # but different bytes
    record_path = release / RECORD
    record = parse_json_text(record_path.read_text(encoding="utf-8"))
    record["call_specs"]["CALL_001"] = text_hash(path.read_text(encoding="utf-8"))  # record updated to match
    record_path.write_text(serialize_json(record), encoding="utf-8", newline="\n")
    assert main(["--dataset-root", str(release), "--validate"]) == 1
    assert "CALL_001.json: not the canonical serialization" in capsys.readouterr().err
    assert main(["--dataset-root", str(release), "--check"]) == 1


def test_reordered_record_is_rejected_by_validate(release, capsys):
    record_path = release / RECORD
    record = parse_json_text(record_path.read_text(encoding="utf-8"))
    record_path.write_text(serialize_json({key: record[key] for key in sorted(record)}), encoding="utf-8", newline="\n")
    assert main(["--dataset-root", str(release), "--validate"]) == 1
    assert "call_spec_record.json: not the canonical serialization" in capsys.readouterr().err


@pytest.mark.parametrize("field,value", [
    ("language", "en-GB"), ("duration_profile", "LONG"), ("target_duration_max_seconds", 301), ("call_intents", ["EXECUTION"]),
])
def test_mutated_metadata_fails_validation(release, field, value):
    _edit_spec(release, "CALL_002", lambda spec: spec["metadata"].update({field: value}))
    assert any("metadata differs from call_manifest.csv" in e for e in _validate(release))
    assert main(["--dataset-root", str(release), "--validate"]) == 1


def test_package_has_no_llm_or_network_imports():
    package = Path(compile_module.__file__).parent
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    for forbidden in ("anthropic", "openai", "boto3", "requests", "httpx", "urllib.request", "socket", "google.genai"):
        assert f"import {forbidden}" not in source and f"from {forbidden}" not in source
