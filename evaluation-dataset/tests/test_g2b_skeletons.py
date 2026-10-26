"""G2B check_plan_skeleton: deterministic structural evidence blueprint compiled from the G2A specs.

Pure compiler tests use the committed call specs; validator mutations run on in-memory texts;
disk-safety tests work on temporary copies. No LLM, network or AWS service is used.
"""

import json
import os
import re
import shutil
import stat
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json, text_hash
from fyp_evaluation.check_plan_skeletons import edge_cases as edge_module
from fyp_evaluation.check_plan_skeletons import model
from fyp_evaluation.check_plan_skeletons.cli import main
from fyp_evaluation.check_plan_skeletons.compile import RECORD, SPEC_DIR, compile_skeleton, skeleton_path
from fyp_evaluation.check_plan_skeletons.edge_cases import COMPILERS, missing_compilers, unknown_compilers
from fyp_evaluation.check_plan_skeletons.inputs import G2AStateError, g0_hashes, load_inputs
from fyp_evaluation.check_plan_skeletons.outputs import check_outputs, generate, read_disk, write_skeletons
from fyp_evaluation.check_plan_skeletons.validate import check_skeleton_schemas, validate_skeleton_set
from fyp_evaluation.validation.stage0 import load_stage0
from release_copy import copy_tree

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
RELEASE = ROOT / "datasets" / "advisory_execution_pilot_v1"
CALL_IDS = [f"CALL_{n:03d}" for n in range(1, 51)]
PATTERNS = {
    **{q: "EMPLOYEE_ONLY" for q in ("C01", "C02", "C04", "C05", "C07", "C11", "C12")},
    **{q: "EXCHANGE" for q in ("C03", "C06", "C08")},
    **{q: "ANY_SPEAKER" for q in ("C09", "C10")},
}


# ---------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def docs():
    return load_stage0(RELEASE)


@pytest.fixture(scope="module")
def generated():
    return generate(RELEASE, SCHEMAS)


@pytest.fixture(scope="module")
def inputs(generated):
    return generated[0]


@pytest.fixture(scope="module")
def texts(generated):
    return generated[1]


@pytest.fixture(scope="module")
def skeletons(texts):
    return {cid: parse_json_text(texts[skeleton_path(cid)]) for cid in CALL_IDS}


@pytest.fixture(scope="module")
def disk():
    return {cid: parse_json_text((RELEASE / skeleton_path(cid)).read_text(encoding="utf-8")) for cid in CALL_IDS}


def _copy_release(source, destination):
    """Copy file contents only (shutil.copytree also copies a OneDrive ReadOnly directory attribute), without rendered audio."""
    copy_tree(source, destination)


def _remove_tree(path):
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


@pytest.fixture()
def release(tmp_path):
    destination = tmp_path / "advisory_execution_pilot_v1"
    destination.mkdir()
    _copy_release(RELEASE, destination)
    return destination


def _snapshot(release):
    files = {p.name: p.read_bytes() for p in (release / SPEC_DIR).iterdir() if p.is_file() and not p.is_symlink()}
    files[RECORD] = (release / RECORD).read_bytes()
    return files


# ---------------------------------------------------------------------------------------- helpers
def evs(sk, aid=None, kind=None):
    return [e for e in sk["planned_events"]
            if (aid is None or aid in e["assignment_ids"]) and (kind is None or e["kind"] == kind)]


def cons(sk, aid=None, ctype=None):
    source = {"type": "EDGE_ASSIGNMENT", "id": aid}
    return [c for c in sk["constraints"] if (aid is None or c["source"] == source) and (ctype is None or c["type"] == ctype)]


def chk(sk, q):
    return next(c for c in sk["checks"] if c["question_id"] == q)


def ev_by_id(sk, event_id):
    return next(e for e in sk["planned_events"] if e["event_id"] == event_id)


def first_with(inputs, *edge_ids):
    for call_id in CALL_IDS:
        present = {e["edge_case_id"] for e in inputs.specs[call_id]["edge_cases"]}
        if set(edge_ids) <= present:
            return call_id
    raise AssertionError(f"no call has {edge_ids}")


def compile_call(inputs, call_id, spec=None):
    spec = spec or inputs.specs[call_id]
    return compile_skeleton(spec, text_hash(serialize_json(spec)), inputs.documents, inputs.g0_hashes)


def edge(inputs, edge_id):
    call_id = first_with(inputs, edge_id)
    return call_id, next(a for a in inputs.specs[call_id]["edge_cases"] if a["edge_case_id"] == edge_id)


# ---------------------------------------------------------------------------------------- contract
def test_schemas_are_valid_and_match_the_python_vocabularies():
    check_skeleton_schemas(SCHEMAS)
    schema = json.loads((SCHEMAS / "stage2" / "check_plan_skeleton.schema.json").read_text(encoding="utf-8"))
    event = schema["$defs"]["event"]["properties"]
    assert tuple(event["kind"]["enum"]) == model.EVENT_KINDS
    assert tuple(event["status"]["enum"]) == model.EVENT_STATUSES
    types = [c["properties"]["type"] for c in schema["$defs"]["constraint"]["oneOf"]]
    assert {t.get("const") for t in types if "const" in t} | {x for t in types for x in t.get("enum", [])} == set(model.CONSTRAINT_TYPES)
    assert set(model.KIND_STATUSES) == set(model.EVENT_KINDS)


def test_every_catalog_edge_case_has_exactly_one_compiler(docs):
    assert missing_compilers(docs["edge_case_catalog"]) == [] and unknown_compilers(docs["edge_case_catalog"]) == []
    assert len(COMPILERS) == len(docs["edge_case_catalog"]["edge_cases"]) == 21


def test_missing_compiler_is_refused(inputs, docs):
    extended = deepcopy(docs)
    clone = deepcopy(next(c for c in extended["edge_case_catalog"]["edge_cases"] if c["id"] == "EC_FALSE_START"))
    clone["id"] = "EC_UNIMPLEMENTED_CASE"
    extended["edge_case_catalog"]["edge_cases"].append(clone)
    with pytest.raises(ValueError, match="EC_UNIMPLEMENTED_CASE"):
        compile_skeleton(inputs.specs["CALL_001"], "0" * 64, extended, inputs.g0_hashes)


def test_exactly_50_skeletons_exist_on_disk():
    assert sorted(p.name for p in (RELEASE / SPEC_DIR).iterdir()) == [f"{cid}.json" for cid in CALL_IDS]
    assert (RELEASE / RECORD).is_file()


def test_top_level_contract(disk, inputs):
    for call_id, sk in disk.items():
        assert list(sk) == ["schema_version", "stage", "dataset_id", "call_id", "call_spec_sha256", "source",
                            "planned_events", "checks", "constraints"]
        assert sk["stage"] == "G2B" and sk["call_id"] == call_id
        assert sk["call_spec_sha256"] == inputs.spec_hashes[call_id]
        assert sk["source"] == inputs.g0_hashes
        assert [e["event_id"] for e in sk["planned_events"]] == [f"EV_{n:03d}" for n in range(1, len(sk["planned_events"]) + 1)]


def test_exactly_20_checks_in_checklist_order_with_spec_answers(disk, inputs):
    checklist_ids = [i["id"] for i in inputs.documents["checklist"]]
    for call_id, sk in disk.items():
        assert [c["question_id"] for c in sk["checks"]] == checklist_ids and len(checklist_ids) == 20
        expected = {g["question_id"]: g["expected"] for g in inputs.specs[call_id]["gold"]}
        for c in sk["checks"]:
            assert c["expected"] == expected[c["question_id"]]
            assert json.dumps(c["expected"]) == json.dumps(expected[c["question_id"]])  # identical JSON types


def test_compliance_modes_refs_patterns_and_scope(disk, inputs):
    items = {i["id"]: i for i in inputs.documents["checklist"]}
    for call_id, sk in disk.items():
        for c in sk["checks"]:
            q = c["question_id"]
            if items[q]["type"] != "COMPLIANCE":
                continue
            verdict = c["expected"]["expected_verdict"]
            requirement = items[q]["requirement_type"]
            presence = (requirement == "OBLIGATION") == (verdict == "PASS")
            ref = f"{q}.{'pass_condition' if requirement == 'OBLIGATION' else 'fail_condition'}"
            assert c["presence_pattern"] == PATTERNS[q]
            assert c["evidence_mode"] == ("PRESENCE" if presence else "ABSENCE")
            if presence:
                assert c["required_condition_ref"] == ref and "forbidden_condition_ref" not in c and "absence" not in c
                assert c["evidence_event_ids"]
            else:
                assert c["forbidden_condition_ref"] == ref and "required_condition_ref" not in c
                assert c["absence"] == {"scope": "ENTIRE_DIALOGUE"} and c["evidence_event_ids"] == []


def test_obligation_prohibition_condition_table(disk):
    seen = set()
    for sk in disk.values():
        for c in sk["checks"]:
            if c["kind"] == "COMPLIANCE":
                key = ("OBLIGATION" if c["question_id"] in {f"C0{n}" for n in range(1, 9)} else "PROHIBITION",
                       c["expected"]["expected_verdict"], c["evidence_mode"],
                       (c.get("required_condition_ref") or c.get("forbidden_condition_ref")).split(".")[1])
                seen.add(key)
    assert seen == {
        ("OBLIGATION", "PASS", "PRESENCE", "pass_condition"), ("OBLIGATION", "FAIL", "ABSENCE", "pass_condition"),
        ("PROHIBITION", "PASS", "ABSENCE", "fail_condition"), ("PROHIBITION", "FAIL", "PRESENCE", "fail_condition"),
    }


def test_presence_patterns_compile_to_the_agreed_events(disk):
    for sk in disk.values():
        for c in sk["checks"]:
            if c["kind"] != "COMPLIANCE" or c["evidence_mode"] != "PRESENCE":
                continue
            acts = [ev_by_id(sk, e) for e in c["evidence_event_ids"]]
            pattern = c["presence_pattern"]
            if pattern == "EXCHANGE":
                assert [(a["allowed_speakers"], a["role"]) for a in acts] == [(["EMPLOYEE"], "ACT"), (["CLIENT"], "RESPONSE")]
                assert any(k["type"] == "ORDER" and (k["before"], k["after"]) == (acts[0]["event_id"], acts[1]["event_id"])
                           and k["source"] == {"type": "PRESENCE_PATTERN", "question_id": c["question_id"]} for k in sk["constraints"])
            elif pattern == "ANY_SPEAKER":
                assert [(a["allowed_speakers"], a["role"]) for a in acts] == [(["CLIENT", "EMPLOYEE"], "ACT")]
            else:
                assert [(a["allowed_speakers"], a["role"]) for a in acts] == [(["EMPLOYEE"], "ACT")]


def test_extraction_checks_have_one_final_event_equal_to_gold(disk, inputs):
    for call_id, sk in disk.items():
        gold = {g["question_id"]: g["expected"]["expected_value"] for g in inputs.specs[call_id]["gold"] if g["expected"]["kind"] == "extraction"}
        assert len(gold) == 8
        for q, value in gold.items():
            c = chk(sk, q)
            assert c["kind"] == "EXTRACTION" and c["resolution_ref"] == f"{q}.resolution_rule" and c["evidence_mode"] == "PRESENCE"
            finals = [e for e in sk["planned_events"] if e["status"] == "FINAL" and q in e["question_ids"]]
            assert [e["event_id"] for e in finals] == [c["final_event_id"]]
            assert type(finals[0]["values"][q]["value"]) is type(value) and finals[0]["values"][q]["value"] == value
            assert c["final_event_id"] in c["evidence_event_ids"]


def test_replacement_calls_share_one_final_event_across_t_questions(disk, inputs):
    calls = [cid for cid in CALL_IDS if any(e["edge_case_id"] == "EC_COMPLETE_ORDER_REPLACEMENT" for e in inputs.specs[cid]["edge_cases"])]
    assert len(calls) == 3
    for cid in calls:
        finals = {chk(disk[cid], q)["final_event_id"] for q in ("T01", "T02", "T03", "T04", "T05", "T06")}
        assert len(finals) == 1
        assert ev_by_id(disk[cid], finals.pop())["kind"] == "ORDER_STATEMENT"


def test_every_edge_assignment_is_represented_and_edges_are_known(disk, inputs):
    total = 0
    for call_id, sk in disk.items():
        represented = {a for e in sk["planned_events"] for a in e["assignment_ids"]} | {a for c in sk["checks"] for a in c["assignment_ids"]}
        assigned = {a["assignment_id"] for a in inputs.specs[call_id]["edge_cases"]}
        assert assigned <= represented and represented <= assigned
        total += len(assigned)
    assert total == 90


def test_references_point_to_existing_events_and_order_is_acyclic(disk):
    for sk in disk.values():
        ids = {e["event_id"] for e in sk["planned_events"]}
        for e in sk["planned_events"]:
            for field in ("corrects_event_id", "invalidates_event_id", "refers_to_event_id"):
                assert e.get(field, next(iter(ids))) in ids
        for c in sk["checks"]:
            assert set(c["evidence_event_ids"]) <= ids
        edges = [(c["before"], c["after"]) for c in sk["constraints"] if c["type"] == "ORDER"]
        for c in sk["constraints"]:
            assert set([c["before"], c["after"]] if c["type"] == "ORDER" else c["events"]) <= ids
        remaining = {i: set() for i in ids}
        for before, after in edges:
            remaining[after].add(before)
        while remaining:  # Kahn: every node must eventually have all predecessors removed
            free = [n for n, preds in remaining.items() if not preds]
            assert free, "ORDER graph has a cycle"
            for n in free:
                del remaining[n]
            for preds in remaining.values():
                preds.difference_update(free)


def test_no_turn_ids_positions_or_wording_anywhere(disk):
    forbidden_keys = re.compile(r"turn_id|turn_index|turn_number|position|dialogue_index|utterance|transcript|wording_text")

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)
    for sk in disk.values():
        assert not [k for k in keys(sk) if forbidden_keys.search(k)]
        assert {c["type"] for c in sk["constraints"]} <= set(model.CONSTRAINT_TYPES)


def test_only_global_absence_no_pairwise_guards(disk):
    for sk in disk.values():
        assert "guards" not in sk and "absence_guards" not in json.dumps(sk)
        for c in sk["checks"]:
            if c["kind"] == "COMPLIANCE" and c["evidence_mode"] == "ABSENCE":
                assert c["absence"] == {"scope": "ENTIRE_DIALOGUE"}


# ---------------------------------------------------------------------------------------- ordering rules
def test_c03_follows_the_c02_disclosure(disk):
    calls = [sk for sk in disk.values() if chk(sk, "C03")["evidence_mode"] == "PRESENCE"]
    assert calls
    for sk in calls:
        c02 = chk(sk, "C02")["evidence_event_ids"]
        c03_act = chk(sk, "C03")["evidence_event_ids"][0]
        rule = {"type": "ORDERING_RULE", "id": "C03_AFTER_C02_DISCLOSURE"}
        assert [k["before"] for k in sk["constraints"] if k["source"] == rule and k["after"] == c03_act] == c02


def test_c03_has_no_ordering_when_it_fails_by_absence(disk):
    for sk in disk.values():
        if chk(sk, "C03")["evidence_mode"] == "ABSENCE":
            assert not [k for k in sk["constraints"] if k["source"].get("id") == "C03_AFTER_C02_DISCLOSURE"]


def test_c08_follows_every_final_transaction_event(disk):
    rule = {"type": "ORDERING_RULE", "id": "C08_AFTER_FINAL_TRANSACTION_STATE"}
    ordered = 0
    for sk in disk.values():
        finals = {chk(sk, q)["final_event_id"] for q in ("T01", "T02", "T03", "T04", "T05", "T06")}
        requests = [e["event_id"] for e in sk["planned_events"] if e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == ["C08"] and e["role"] == "ACT"]
        got = {(k["before"], k["after"]) for k in sk["constraints"] if k["source"] == rule}
        assert got == {(f, r) for f in finals for r in requests}
        ordered += bool(got)
    assert ordered >= 40  # C08 PASS calls plus the weak-confirmation requests


def test_no_other_ordering_rule_exists(disk):
    ids = {k["source"]["id"] for sk in disk.values() for k in sk["constraints"] if k["source"]["type"] == "ORDERING_RULE"}
    assert ids == {"C03_AFTER_C02_DISCLOSURE", "C08_AFTER_FINAL_TRANSACTION_STATE"}


# ---------------------------------------------------------------------------------------- spoken forms
def test_gold_restatement_policy_is_consistent(disk, inputs):
    for call_id, sk in disk.items():
        for c in sk["checks"]:
            if c["kind"] != "EXTRACTION":
                continue
            q = c["question_id"]
            gold = c["expected"]["expected_value"]
            for e in sk["planned_events"]:
                entry = e.get("values", {}).get(q)
                if entry is None:
                    continue
                if e["kind"] == "REFERENCE_SELECTION":
                    assert entry["spoken_form"] == {"requirement": "NOT_EXPLICIT"}
                elif entry["value"] == gold:
                    assert entry["spoken_form"] == c["gold_restatement"]["spoken_form"], (call_id, e["event_id"])
                else:
                    assert entry["spoken_form"] == c["non_gold_spoken_form"]
                    assert entry["spoken_form"]["requirement"] == "ONE_OF"  # a non-gold value never uses an edge-only form
                    assert e["status"] not in {"FINAL", "STATED"}


def test_spoken_edge_requires_the_special_form_and_forbids_the_others(disk, inputs):
    required = {"EC_OH_VS_ZERO": ("K01", "OH_FOR_ZERO", "ALL_ZERO_DIGITS"),
                "EC_DOUBLE_TRIPLE_DIGITS": ("K01", "DOUBLE_OR_TRIPLE_DIGIT", "ALL_RUNS_OF_ASSIGNED_VARIANT_LENGTH"),
                "EC_FINANCIAL_SHORTHAND": ("T03", "FINANCIAL_SHORTHAND", "WHOLE_VALUE"),
                "EC_LARGE_NUMBER_EXPRESSION": ("T03", "MAGNITUDE_EXPRESSION", "WHOLE_VALUE")}
    seen = 0
    for call_id, sk in disk.items():
        for a in inputs.specs[call_id]["edge_cases"]:
            if a["edge_case_id"] not in required:
                continue
            seen += 1
            q, form, applies = required[a["edge_case_id"]]
            c = chk(sk, q)
            spoken = c["gold_restatement"]["spoken_form"]
            assert c["gold_restatement"]["policy"] == "ASSIGNED_FORM_ONLY"
            assert (spoken["requirement"], spoken["form"], spoken["applies_to"]) == ("REQUIRED", form, applies)
            if "variant" in a["parameters"]:
                assert spoken["variant"] == a["parameters"]["variant"]
            if "spoken_form" in a["parameters"]:
                assert spoken["text"] == a["parameters"]["spoken_form"]
            assert form not in c["forbidden_spoken_forms"]
            assert ev_by_id(sk, c["final_event_id"])["values"][q]["spoken_form"] == spoken
    assert seen == 16


def test_calls_without_a_spoken_edge_forbid_every_special_form(disk, inputs):
    for call_id, sk in disk.items():
        edges = {a["edge_case_id"] for a in inputs.specs[call_id]["edge_cases"]}
        k01, t03 = chk(sk, "K01"), chk(sk, "T03")
        if not edges & {"EC_OH_VS_ZERO", "EC_DOUBLE_TRIPLE_DIGITS"}:
            assert k01["forbidden_spoken_forms"] == ["OH_FOR_ZERO", "DOUBLE_OR_TRIPLE_DIGIT"]
            assert k01["gold_restatement"]["policy"] == "ANY_PERMITTED_FORM"
        if not edges & {"EC_FINANCIAL_SHORTHAND", "EC_LARGE_NUMBER_EXPRESSION"}:
            assert t03["forbidden_spoken_forms"] == ["FINANCIAL_SHORTHAND", "MAGNITUDE_EXPRESSION"]
            assert t03["gold_restatement"]["policy"] == "ANY_PERMITTED_FORM"
        for c in sk["checks"]:
            if c["kind"] == "EXTRACTION" and c["gold_restatement"]["policy"] == "ANY_PERMITTED_FORM":
                assert c["gold_restatement"]["spoken_form"]["requirement"] == "ONE_OF"


def test_call_010_applies_500k_to_both_unchanged_t03_mentions(disk, inputs):
    sk = disk["CALL_010"]
    orders = [e for e in sk["planned_events"] if e["kind"] == "ORDER_STATEMENT"]
    assert [(e["status"], e["values"]["T03"]["value"]) for e in orders] == [("SUPERSEDED", 500000), ("FINAL", 500000)]
    for e in orders:
        assert e["values"]["T03"]["spoken_form"] == {"requirement": "REQUIRED", "form": "FINANCIAL_SHORTHAND",
                                                      "applies_to": "WHOLE_VALUE", "text": "500k"}
    changed = {q for q in ("T01", "T02", "T04", "T05", "T06")
               if orders[0]["values"][q]["value"] != orders[1]["values"][q]["value"]}
    assert changed == {"T01", "T04"}
    assert orders[0]["values"]["T04"]["spoken_form"]["requirement"] == "ONE_OF"  # a genuinely superseded value stays ordinary
    assert chk(sk, "T03")["forbidden_spoken_forms"] == ["MAGNITUDE_EXPRESSION"]
    assert not [e for e in sk["planned_events"] if e["status"] == "SUPERSEDED" and "T03" in e.get("values", {}) and e["kind"] != "ORDER_STATEMENT"]


# ---------------------------------------------------------------------------------------- one fixture test per edge case
def _same_turn(sk, a, spec):
    q, p, aid = a["target_question_ids"][0], a["parameters"], a["assignment_id"]
    first, final = evs(sk, aid)
    assert (first["kind"], first["status"], first["values"][q]["value"]) == ("VALUE_STATEMENT", "SUPERSEDED", p["initial_value"])
    assert (final["kind"], final["status"], final["values"][q]["value"]) == ("VALUE_CORRECTION", "FINAL", p["final_value"])
    assert final["corrects_event_id"] == first["event_id"] and p["initial_value"] != p["final_value"]
    assert [c["type"] for c in cons(sk, aid)] == ["ORDER", "SAME_TURN"]


def _cross_turn(sk, a, spec):
    q, p, aid = a["target_question_ids"][0], a["parameters"], a["assignment_id"]
    first, final = evs(sk, aid)
    assert (first["status"], first["values"][q]["value"]) == ("SUPERSEDED", p["initial_value"])
    assert (final["status"], final["values"][q]["value"]) == ("FINAL", p["final_value"])
    assert sorted(c["type"] for c in cons(sk, aid)) == ["MIN_INTERVENING_TURNS", "NOT_SAME_TURN", "ORDER"]
    assert cons(sk, aid, "MIN_INTERVENING_TURNS")[0]["turns"] == 2


def _multiple_corrections(sk, a, spec):
    q, p, aid = a["target_question_ids"][0], a["parameters"], a["assignment_id"]
    events = evs(sk, aid)
    assert [e["status"] for e in events] == ["SUPERSEDED"] * len(p["superseded_values"]) + ["FINAL"]
    assert [e["values"][q]["value"] for e in events] == p["superseded_values"] + [p["final_value"]]
    assert len(p["superseded_values"]) >= 2
    assert [e.get("corrects_event_id") for e in events[1:]] == [e["event_id"] for e in events[:-1]]
    assert sorted(c["type"] for c in cons(sk, aid)) == ["ORDER"] * len(events[1:])  # ordered chain, no gap, no turn rule


def _readback(minimum):
    def check(sk, a, spec):
        q, p, aid = a["target_question_ids"][0], a["parameters"], a["assignment_id"]
        stated, readback, final = evs(sk, aid)
        assert (stated["kind"], stated["status"], stated["allowed_speakers"]) == ("VALUE_STATEMENT", "STATED", ["CLIENT"])
        assert stated["values"][q]["value"] == p["final_value"]
        assert (readback["kind"], readback["status"], readback["allowed_speakers"]) == ("READBACK", "REJECTED", ["EMPLOYEE"])
        rb = readback["readback"]
        assert (rb["must_include_question_ids"], rb["min_field_count"], rb["wrong_question_id"], rb["wrong_value"]) == (
            [q], minimum, q, p["employee_wrong_value"])
        assert rb["other_included_fields_must_match_final"] is True
        assert rb["other_included_fields_must_follow_their_final_events"] is True
        assert (final["status"], final["values"][q]["value"], final["corrects_event_id"]) == ("FINAL", p["final_value"], readback["event_id"])
        assert [(c["before"], c["after"]) for c in cons(sk, aid, "ORDER")] == [
            (stated["event_id"], readback["event_id"]), (readback["event_id"], final["event_id"])]
    return check


def _order_replacement(sk, a, spec):
    p, aid = a["parameters"], a["assignment_id"]
    initial, invalidation, final = evs(sk, aid)
    assert [(e["kind"], e["status"]) for e in (initial, invalidation, final)] == [
        ("ORDER_STATEMENT", "SUPERSEDED"), ("ORDER_INVALIDATION", "CONTEXT"), ("ORDER_STATEMENT", "FINAL")]
    t_ids = ["T01", "T02", "T03", "T04", "T05", "T06"]
    assert set(initial["values"]) == set(final["values"]) == set(t_ids) and invalidation["invalidates_event_id"] == initial["event_id"]
    assert {q: initial["values"][q]["value"] for q in t_ids} == p["initial_order"]
    assert {q: final["values"][q]["value"] for q in t_ids} == p["final_order"]
    assert [q for q in t_ids if p["initial_order"][q] != p["final_order"][q]] == a["target_question_ids"]
    assert {chk(sk, q)["final_event_id"] for q in t_ids} == {final["event_id"]}


def _currency_distractor(sk, a, spec):
    q, p, aid = a["target_question_ids"][0], a["parameters"], a["assignment_id"]
    (d,) = evs(sk, aid)
    assert (d["kind"], d["status"], d["allowed_speakers"]) == ("DISTRACTOR_MENTION", "DISTRACTOR", ["CLIENT", "EMPLOYEE"])
    assert d["values"][q]["value"] == p["distractor_currency"] != p["final_currency"] and d["distractor_role"] == p["distractor_role"]
    assert chk(sk, q)["final_event_id"] != d["event_id"] and ev_by_id(sk, chk(sk, q)["final_event_id"])["values"][q]["value"] == p["final_currency"]


def _monetary_distractors(sk, a, spec):
    q, p, aid = a["target_question_ids"][0], a["parameters"], a["assignment_id"]
    distractors = evs(sk, aid)
    assert [(e["values"][q]["value"], e["distractor_role"], e["currency"]) for e in distractors] == [
        (d["amount"], d["role"], d["currency"]) for d in p["distractor_values"]]
    assert all(e["status"] == "DISTRACTOR" and e["allowed_speakers"] == ["CLIENT", "EMPLOYEE"] for e in distractors)
    assert ev_by_id(sk, chk(sk, q)["final_event_id"])["values"][q]["value"] == p["final_transaction_amount"]
    assert all(e["event_id"] != chk(sk, q)["final_event_id"] for e in distractors)


def _value_reference(parameter):
    def check(sk, a, spec):
        q, p, aid = a["target_question_ids"][0], a["parameters"], a["assignment_id"]
        antecedent, selection = evs(sk, aid)
        assert (antecedent["kind"], antecedent["status"], antecedent["allowed_speakers"]) == (
            "ANTECEDENT_MENTION", "ANTECEDENT", [p["antecedent_speaker"]])
        assert antecedent["values"][q]["value"] == p[parameter]
        assert (selection["kind"], selection["status"], selection["allowed_speakers"]) == ("REFERENCE_SELECTION", "FINAL", ["CLIENT"])
        assert selection["values"][q] == {"value": p[parameter], "spoken_form": {"requirement": "NOT_EXPLICIT"}}
        assert selection["refers_to_event_id"] == antecedent["event_id"]
        assert [(c["before"], c["after"]) for c in cons(sk, aid, "ORDER")] == [(antecedent["event_id"], selection["event_id"])]
    return check


def _product_reference(sk, a, spec):
    q, p, aid = a["target_question_ids"][0], a["parameters"], a["assignment_id"]
    antecedents = evs(sk, aid, "ANTECEDENT_MENTION")
    assert [e["values"][q]["value"] for e in antecedents] == p["candidate_products"]  # locked mention order, never reshuffled
    assert all(e["allowed_speakers"] == [p["antecedent_speaker"]] for e in antecedents)
    (selection,) = evs(sk, aid, "REFERENCE_SELECTION")
    chosen = antecedents[p["candidate_products"].index(p["selected_product"])]
    assert selection["refers_to_event_id"] == chosen["event_id"] and selection["values"][q]["value"] == p["selected_product"]
    chain = [(c["before"], c["after"]) for c in cons(sk, aid, "ORDER")]
    assert chain == [(x["event_id"], y["event_id"]) for x, y in zip(antecedents, antecedents[1:])] + [(antecedents[-1]["event_id"], selection["event_id"])]


def _spoken(q, form, extra=None):
    def check(sk, a, spec):
        aid, c = a["assignment_id"], chk(sk, q)
        spoken = c["gold_restatement"]["spoken_form"]
        assert c["gold_restatement"]["policy"] == "ASSIGNED_FORM_ONLY" and spoken["requirement"] == "REQUIRED" and spoken["form"] == form
        for key, parameter in (extra or {}).items():
            assert spoken[key] == a["parameters"][parameter]
        final = ev_by_id(sk, c["final_event_id"])
        assert final["values"][q]["spoken_form"] == spoken and aid in final["assignment_ids"]
        assert not [e for e in sk["planned_events"] if aid in e["assignment_ids"] and e["event_id"] != final["event_id"]
                    and e["values"][q]["spoken_form"]["requirement"] != "REQUIRED"]
        assert c["forbidden_spoken_forms"] and form not in c["forbidden_spoken_forms"]
    return check


def _false_start(sk, a, spec):
    q, aid = a["target_question_ids"][0], a["assignment_id"]
    (fragment,) = evs(sk, aid)
    assert (fragment["kind"], fragment["status"], fragment["incomplete"], fragment["allowed_speakers"]) == ("FALSE_START", "ABANDONED", True, ["CLIENT"])
    assert "values" not in fragment  # no complete value, no invented text
    final = chk(sk, q)["final_event_id"]
    assert [(c["type"], c.get("before"), c.get("after"), c.get("events")) for c in cons(sk, aid)] == [
        ("ORDER", fragment["event_id"], final, None), ("SAME_TURN", None, None, sorted([fragment["event_id"], final]))]


def _long_distance(sk, a, spec):
    aid = a["assignment_id"]
    finals = sorted(chk(sk, q)["final_event_id"] for q in a["target_question_ids"])
    assert len(set(finals)) == 2 and evs(sk, aid) == []
    (gap,) = cons(sk, aid)
    assert (gap["type"], gap["events"], gap["turns"]) == ("MIN_INTERVENING_TURNS", finals, 6)  # unordered pair, six complete turns
    assert "before" not in gap


def _guarantee(sk, a, spec):
    aid = a["assignment_id"]
    request, refusal = evs(sk, aid)
    assert (request["kind"], request["allowed_speakers"], refusal["kind"], refusal["allowed_speakers"]) == (
        "GUARANTEE_REQUEST", ["CLIENT"], "GUARANTEE_REFUSAL", ["EMPLOYEE"])
    assert [(c["before"], c["after"]) for c in cons(sk, aid, "ORDER")] == [(request["event_id"], refusal["event_id"])]
    c12 = chk(sk, "C12")
    assert (c12["evidence_mode"], c12["forbidden_condition_ref"]) == ("ABSENCE", "C12.fail_condition")  # the refusal never creates the prohibited conduct


def _paraphrase(sk, a, spec):
    q, aid = a["target_question_ids"][0], a["assignment_id"]
    c = chk(sk, q)
    assert c["wording_constraint"] == "NOT_VERBATIM_CHECKLIST" and c["evidence_mode"] == "PRESENCE"
    assert [e["event_id"] for e in evs(sk, aid)] == c["evidence_event_ids"]
    assert not re.search(r"ngram|threshold|overlap|distance|similarity", json.dumps(sk), re.I)  # symbolic rule only


def _risk(sk, a, spec):
    aid = a["assignment_id"]
    assert (chk(sk, "C02")["evidence_mode"], chk(sk, "C03")["evidence_mode"]) == ("PRESENCE", "ABSENCE")
    assert [e["event_id"] for e in evs(sk, aid)] == chk(sk, "C02")["evidence_event_ids"]
    assert chk(sk, "C03")["evidence_event_ids"] == [] and not [e for e in sk["planned_events"] if "C03" in e["question_ids"]]


def _weak(sk, a, spec):
    aid = a["assignment_id"]
    request, weak = evs(sk, aid)
    assert (request["kind"], request["role"], request["allowed_speakers"]) == ("COMPLIANCE_ACT", "ACT", ["EMPLOYEE"])
    assert (weak["kind"], weak["strength"], weak["allowed_speakers"]) == ("WEAK_ACKNOWLEDGEMENT", "WEAK", ["CLIENT"])
    c08 = chk(sk, "C08")
    assert c08["evidence_mode"] == "ABSENCE" and c08["evidence_event_ids"] == []  # the complete exchange stays absent
    assert not [e for e in sk["planned_events"] if e["kind"] == "COMPLIANCE_ACT" and e["role"] == "RESPONSE" and e["question_ids"] == ["C08"]]
    assert [(c["before"], c["after"]) for c in cons(sk, aid, "ORDER")] == [(request["event_id"], weak["event_id"])]


EDGE_FIXTURES = {
    "EC_CORRECTION_SAME_TURN": _same_turn,
    "EC_CORRECTION_CROSS_TURN": _cross_turn,
    "EC_MULTIPLE_CORRECTIONS": _multiple_corrections,
    "EC_EMPLOYEE_READBACK_ERROR": _readback(1),
    "EC_ONE_FIELD_READBACK_ERROR": _readback(2),
    "EC_COMPLETE_ORDER_REPLACEMENT": _order_replacement,
    "EC_MULTIPLE_CURRENCIES": _currency_distractor,
    "EC_MULTIPLE_MONETARY_VALUES": _monetary_distractors,
    "EC_PREVIOUS_CURRENCY_REFERENCE": _value_reference("final_currency"),
    "EC_PRODUCT_REFERENCE": _product_reference,
    "EC_PREVIOUS_QUANTITY_REFERENCE": _value_reference("final_quantity"),
    "EC_DOUBLE_TRIPLE_DIGITS": _spoken("K01", "DOUBLE_OR_TRIPLE_DIGIT", {"variant": "variant"}),
    "EC_OH_VS_ZERO": _spoken("K01", "OH_FOR_ZERO"),
    "EC_FINANCIAL_SHORTHAND": _spoken("T03", "FINANCIAL_SHORTHAND", {"text": "spoken_form"}),
    "EC_LARGE_NUMBER_EXPRESSION": _spoken("T03", "MAGNITUDE_EXPRESSION", {"text": "spoken_form"}),
    "EC_FALSE_START": _false_start,
    "EC_LONG_DISTANCE_DEPENDENCY": _long_distance,
    "EC_GUARANTEE_REQUEST_REFUSAL": _guarantee,
    "EC_PARAPHRASED_COMPLIANCE_STATEMENT": _paraphrase,
    "EC_RISK_EXPLAINED_UNDERSTANDING_NOT_VERIFIED": _risk,
    "EC_WEAK_CONFIRMATION": _weak,
}


def test_there_is_one_fixture_test_per_catalog_edge_case(docs):
    assert set(EDGE_FIXTURES) == {c["id"] for c in docs["edge_case_catalog"]["edge_cases"]}


@pytest.mark.parametrize("edge_id", sorted(EDGE_FIXTURES))
def test_edge_case_fixture(inputs, edge_id):
    call_id, assignment = edge(inputs, edge_id)
    spec = inputs.specs[call_id]
    sk = compile_call(inputs, call_id)
    EDGE_FIXTURES[edge_id](sk, assignment, spec)


@pytest.mark.parametrize("edge_id", sorted(EDGE_FIXTURES))
def test_edge_case_fixture_holds_for_every_occurrence(inputs, skeletons, edge_id):
    for call_id in CALL_IDS:
        for a in inputs.specs[call_id]["edge_cases"]:
            if a["edge_case_id"] == edge_id:
                EDGE_FIXTURES[edge_id](skeletons[call_id], a, inputs.specs[call_id])


def test_one_field_readback_leaves_optional_fields_unresolved(skeletons, inputs):
    sk = skeletons[first_with(inputs, "EC_ONE_FIELD_READBACK_ERROR")]
    (readback,) = [e for e in sk["planned_events"] if e["kind"] == "READBACK"]
    q = readback["readback"]["wrong_question_id"]
    assert readback["question_ids"] == [q] and readback["readback"]["must_include_question_ids"] == [q]
    assert "included_question_ids" not in readback["readback"] and readback["readback"]["min_field_count"] == 2


# ---------------------------------------------------------------------------------------- impossible combinations
def test_frozen_dataset_has_no_long_distance_conflict(inputs):
    for call_id in CALL_IDS:
        edges = inputs.specs[call_id]["edge_cases"]
        if any(e["edge_case_id"] == "EC_LONG_DISTANCE_DEPENDENCY" for e in edges):
            assert not any(e["edge_case_id"] == "EC_COMPLETE_ORDER_REPLACEMENT" for e in edges)
        compile_call(inputs, call_id)  # every frozen call compiles without a conflict


def test_long_distance_on_a_shared_final_event_fails_loudly(inputs):
    call_id = first_with(inputs, "EC_COMPLETE_ORDER_REPLACEMENT")
    spec = deepcopy(inputs.specs[call_id])
    spec["edge_cases"].append({
        "assignment_id": "EA_999", "edge_case_id": "EC_LONG_DISTANCE_DEPENDENCY", "family": "DIALOGUE_STRUCTURE",
        "target_question_ids": ["T01", "T02"], "parameters": {"min_turn_gap": 6, "target_question_ids": ["T01", "T02"]}})
    with pytest.raises(ValueError, match="unsatisfiable long-distance dependency"):
        compile_call(inputs, call_id, spec)


def test_supplier_on_a_question_already_finalised_by_replacement_fails(inputs):
    """A reference on T04 cannot coexist with an order replacement that already supplies T04's FINAL event."""
    call_id = next(c for c in CALL_IDS for a in inputs.specs[c]["edge_cases"]
                   if a["edge_case_id"] == "EC_COMPLETE_ORDER_REPLACEMENT" and "T04" not in a["target_question_ids"])
    spec = deepcopy(inputs.specs[call_id])
    quantity = next(g for g in spec["gold"] if g["question_id"] == "T04")["expected"]["expected_value"]
    spec["edge_cases"].append({
        "assignment_id": "EA_999", "edge_case_id": "EC_PREVIOUS_QUANTITY_REFERENCE", "family": "REFERENCE_RESOLUTION",
        "target_question_ids": ["T04"], "parameters": {"antecedent_speaker": "CLIENT", "final_quantity": quantity}})
    with pytest.raises(ValueError, match="already has a FINAL event"):
        compile_call(inputs, call_id, spec)


@pytest.mark.parametrize("change,message", [
    (lambda s: s["edge_cases"][0]["parameters"].update(final_value="CONTRADICTS"), "contradicts frozen gold"),
    (lambda s: s["edge_cases"][0]["parameters"].update(initial_value="PLACEHOLDER"), "equals gold but must differ"),
])
def test_edge_parameters_that_contradict_gold_fail(inputs, change, message):
    call_id = first_with(inputs, "EC_CORRECTION_SAME_TURN")
    spec = deepcopy(inputs.specs[call_id])
    spec["edge_cases"] = [a for a in spec["edge_cases"] if a["edge_case_id"] == "EC_CORRECTION_SAME_TURN"]
    q = spec["edge_cases"][0]["target_question_ids"][0]
    gold = next(g for g in spec["gold"] if g["question_id"] == q)["expected"]["expected_value"]
    if message.startswith("equals"):
        spec["edge_cases"][0]["parameters"]["initial_value"] = gold
    else:
        change(spec)
    with pytest.raises(ValueError, match=message):
        compile_call(inputs, call_id, spec)


def test_unknown_edge_case_is_refused(inputs):
    spec = deepcopy(inputs.specs["CALL_001"])
    spec["edge_cases"].append({"assignment_id": "EA_999", "edge_case_id": "EC_NOT_IN_CATALOG", "family": "X",
                               "target_question_ids": ["T01"], "parameters": {}})
    with pytest.raises(ValueError, match="not in the catalog"):
        compile_call(inputs, "CALL_001", spec)


# ---------------------------------------------------------------------------------------- validator mutations
def _validate_mutated(generated, call_id, change, *, fix_record=True):
    inputs, texts, _ = generated
    texts = dict(texts)
    path = skeleton_path(call_id)
    sk = parse_json_text(texts[path])
    change(sk)
    texts[path] = serialize_json(sk)
    if fix_record:
        record = parse_json_text(texts[RECORD])
        record["skeletons"][call_id]["skeleton_sha256"] = text_hash(texts[path])
        texts[RECORD] = serialize_json(record)
    return validate_skeleton_set(inputs, texts, SCHEMAS)


def _event(sk, kind=None, status=None, q=None, **match):
    for e in sk["planned_events"]:
        if (kind is None or e["kind"] == kind) and (status is None or e["status"] == status) and (q is None or q in e["question_ids"]):
            return e
    raise AssertionError("no event")


def _plain_call(inputs):
    return next(c for c in CALL_IDS if not inputs.specs[c]["edge_cases"] and all(
        g["expected"].get("expected_verdict") != "FAIL" for g in inputs.specs[c]["gold"]))


def _no_spoken(inputs):
    return next(c for c in CALL_IDS if not {e["edge_case_id"] for e in inputs.specs[c]["edge_cases"]} & {
        "EC_OH_VS_ZERO", "EC_DOUBLE_TRIPLE_DIGITS", "EC_FINANCIAL_SHORTHAND", "EC_LARGE_NUMBER_EXPRESSION"})


def _c08_pass(inputs):
    return next(c for c in CALL_IDS if any(g["question_id"] == "C08" and g["expected"]["expected_verdict"] == "PASS" for g in inputs.specs[c]["gold"]))


def _swap_first_two_antecedents(sk):
    first, second = [e for e in sk["planned_events"] if e["kind"] == "ANTECEDENT_MENTION"][:2]
    first["values"], second["values"] = second["values"], first["values"]


def _drop_assignment(sk, aid):
    for e in sk["planned_events"]:
        e["assignment_ids"] = [a for a in e["assignment_ids"] if a != aid]
    for c in sk["checks"]:
        c["assignment_ids"] = [a for a in c["assignment_ids"] if a != aid]


def _add_reverse_order(sk):
    existing = next(c for c in sk["constraints"] if c["type"] == "ORDER" and c["source"]["type"] == "EDGE_ASSIGNMENT")
    sk["constraints"].append({"type": "ORDER", "before": existing["after"], "after": existing["before"], "source": existing["source"]})


MUTATIONS = {
    "wrong required_condition_ref": (lambda i: _plain_call(i), lambda sk: chk(sk, "C01").update(required_condition_ref="C01.fail_condition"), "required_condition_ref must be"),
    "wrong presence_pattern": (lambda i: _plain_call(i), lambda sk: chk(sk, "C01").update(presence_pattern="EXCHANGE"), "presence_pattern"),
    "wrong resolution_ref": (lambda i: _plain_call(i), lambda sk: chk(sk, "T04").update(resolution_ref="T03.resolution_rule"), "resolution_ref must be"),
    "expected answer changed": (lambda i: _plain_call(i), lambda sk: chk(sk, "K02")["expected"].update(expected_value="01 January 2000"), "expected answer differs"),
    "no FINAL event": (lambda i: _plain_call(i), lambda sk: _event(sk, "VALUE_STATEMENT", "FINAL", "K02").update(status="STATED"), "exactly one FINAL event"),
    "two FINAL events": (lambda i: first_with(i, "EC_CORRECTION_SAME_TURN"), lambda sk: _event(sk, "VALUE_STATEMENT", "SUPERSEDED").update(status="FINAL"), "exactly one FINAL event"),
    "FINAL value changed": (lambda i: _plain_call(i), lambda sk: _event(sk, "VALUE_STATEMENT", "FINAL", "K01")["values"]["K01"].update(value="S0000000A"), "differs from gold"),
    "superseded value equals gold": (lambda i: first_with(i, "EC_CORRECTION_SAME_TURN"), lambda sk: (lambda e, g: e["values"][next(iter(e["values"]))].update(value=g))(
        _event(sk, "VALUE_STATEMENT", "SUPERSEDED"), chk(sk, next(iter(_event(sk, "VALUE_STATEMENT", "SUPERSEDED")["values"])))["expected"]["expected_value"]), "equals gold"),
    "distractor equals gold": (lambda i: first_with(i, "EC_MULTIPLE_CURRENCIES"), lambda sk: (lambda e: e["values"]["T02"].update(value=chk(sk, "T02")["expected"]["expected_value"]))(_event(sk, "DISTRACTOR_MENTION")), "equals gold"),
    "readback wrong value equals gold": (lambda i: first_with(i, "EC_EMPLOYEE_READBACK_ERROR"), lambda sk: (lambda e: e["readback"].update(wrong_value=chk(sk, e["readback"]["wrong_question_id"])["expected"]["expected_value"]))(_event(sk, "READBACK")), "readback wrong value must differ"),
    "required form without an edge case": (lambda i: _no_spoken(i), lambda sk: _event(sk, "VALUE_STATEMENT", "FINAL", "K01")["values"]["K01"].update(spoken_form={"requirement": "REQUIRED", "form": "OH_FOR_ZERO", "applies_to": "ALL_ZERO_DIGITS"}), "required spoken form"),
    "default form on an edge-targeted gold mention": (lambda i: "CALL_010", lambda sk: _event(sk, "ORDER_STATEMENT", "SUPERSEDED")["values"]["T03"].update(spoken_form={"requirement": "ONE_OF", "forms": ["DIGITS_WITH_THOUSANDS_SEPARATORS"]}), "gold_restatement form"),
    "forbidden forms emptied": (lambda i: _plain_call(i), lambda sk: chk(sk, "K01").update(forbidden_spoken_forms=[]), "forbidden_spoken_forms must be"),
    "non-gold value with a foreign form": (lambda i: first_with(i, "EC_CORRECTION_SAME_TURN"), lambda sk: (lambda e: e["values"][next(iter(e["values"]))].update(spoken_form={"requirement": "ONE_OF", "forms": ["SOMETHING_ELSE"]}))(_event(sk, "VALUE_STATEMENT", "SUPERSEDED")), "default/permitted forms"),
    "missing MIN_INTERVENING in cross-turn": (lambda i: first_with(i, "EC_CORRECTION_CROSS_TURN"), lambda sk: sk.update(constraints=[c for c in sk["constraints"] if c["type"] != "MIN_INTERVENING_TURNS"]), "missing MIN_INTERVENING_TURNS 2"),
    "missing SAME_TURN in same-turn correction": (lambda i: first_with(i, "EC_CORRECTION_SAME_TURN"), lambda sk: sk.update(constraints=[c for c in sk["constraints"] if c["type"] != "SAME_TURN"]), "missing SAME_TURN"),
    "ORDER cycle": (lambda i: first_with(i, "EC_CORRECTION_SAME_TURN"), _add_reverse_order, "cycle"),
    "SAME_TURN contradicts NOT_SAME_TURN": (lambda i: first_with(i, "EC_CORRECTION_CROSS_TURN"), lambda sk: (lambda c: sk["constraints"].append({"type": "SAME_TURN", "events": c["events"], "source": c["source"]}))(next(c for c in sk["constraints"] if c["type"] == "NOT_SAME_TURN")), "contradicts SAME_TURN"),
    "constraint on a missing event": (lambda i: first_with(i, "EC_CORRECTION_SAME_TURN"), lambda sk: sk["constraints"].append({"type": "SAME_TURN", "events": ["EV_001", "EV_999"], "source": sk["constraints"][0]["source"]}), "distinct existing events"),
    "C08 ordering dropped": (lambda i: _c08_pass(i), lambda sk: sk.update(constraints=[c for c in sk["constraints"] if c["source"].get("id") != "C08_AFTER_FINAL_TRANSACTION_STATE"]), "ordering-rule constraints differ"),
    "invented ordering rule constraint": (lambda i: _plain_call(i), lambda sk: sk["constraints"].append({"type": "ORDER", "before": "EV_001", "after": "EV_002", "source": {"type": "ORDERING_RULE", "id": "C03_AFTER_C02_DISCLOSURE"}}), "ordering-rule constraints differ"),
    "long-distance gap changed": (lambda i: first_with(i, "EC_LONG_DISTANCE_DEPENDENCY"), lambda sk: next(c for c in sk["constraints"] if c["type"] == "MIN_INTERVENING_TURNS").update(turns=5), "missing MIN_INTERVENING_TURNS 6"),
    "product candidate order swapped": (lambda i: first_with(i, "EC_PRODUCT_REFERENCE"), _swap_first_two_antecedents, "locked order"),
    "antecedent speaker changed": (lambda i: first_with(i, "EC_PREVIOUS_CURRENCY_REFERENCE"), lambda sk: _event(sk, "ANTECEDENT_MENTION").update(allowed_speakers=["CLIENT"] if _event(sk, "ANTECEDENT_MENTION")["allowed_speakers"] == ["EMPLOYEE"] else ["EMPLOYEE"]), "antecedent speaker"),
    "distractor speakers narrowed": (lambda i: first_with(i, "EC_MULTIPLE_CURRENCIES"), lambda sk: _event(sk, "DISTRACTOR_MENTION").update(allowed_speakers=["CLIENT"]), "distractor speakers"),
    "assignment unrepresented": (lambda i: first_with(i, "EC_GUARANTEE_REQUEST_REFUSAL"), lambda sk: _drop_assignment(sk, next(a for a in sk["checks"] if a["question_id"] == "C12")["assignment_ids"][0]), "not represented"),
    "unknown assignment referenced": (lambda i: _plain_call(i), lambda sk: sk["planned_events"][0]["assignment_ids"].append("EA_999"), "missing from the call spec"),
    "paraphrase wording constraint dropped": (lambda i: first_with(i, "EC_PARAPHRASED_COMPLIANCE_STATEMENT"), lambda sk: next(c for c in sk["checks"] if "wording_constraint" in c).pop("wording_constraint"), "wording_constraint"),
    "stale call_spec_sha256": (lambda i: _plain_call(i), lambda sk: sk.update(call_spec_sha256="0" * 64), "call_spec_sha256 differs"),
    "stale G0 source hash": (lambda i: _plain_call(i), lambda sk: sk["source"].update(normalization_sha256="0" * 64), "source hashes differ"),
    "incompatible kind and status": (lambda i: _plain_call(i), lambda sk: _event(sk, "COMPLIANCE_ACT").update(status="FINAL"), "cannot have status"),
    "exchange speakers swapped": (lambda i: _c08_pass(i), lambda sk: next(e for e in sk["planned_events"] if e["kind"] == "COMPLIANCE_ACT" and e["role"] == "ACT" and e["question_ids"] == ["C08"]).update(allowed_speakers=["CLIENT"]), "must be exactly"),
    "turn_id injected": (lambda i: _plain_call(i), lambda sk: sk["planned_events"][0].update(turn_id=3), None),
    "pairwise guards injected": (lambda i: _plain_call(i), lambda sk: sk.update(guards=[{"presence_question": "C06", "absence_question": "C05"}]), None),
    "constraint type outside the closed set": (lambda i: _plain_call(i), lambda sk: sk["constraints"].append({"type": "TURN_INDEX", "events": ["EV_001", "EV_002"], "source": {"type": "PRESENCE_PATTERN", "question_id": "C06"}}), None),
    "absence scope changed": (lambda i: _plain_call(i), lambda sk: next(c for c in sk["checks"] if c.get("absence")).update(absence={"scope": "ONE_TURN"}), None),
}


@pytest.mark.parametrize("label", MUTATIONS)
def test_validator_rejects_mutation(generated, label):
    pick, change, message = MUTATIONS[label]
    call_id = pick(generated[0])
    errors = _validate_mutated(generated, call_id, change)
    assert errors, "mutation was accepted"
    if message:
        assert any(message in e for e in errors), errors


def test_clean_skeletons_validate(generated):
    inputs, texts, _ = generated
    assert validate_skeleton_set(inputs, texts, SCHEMAS) == []


@pytest.mark.parametrize("reorder", ["top_level", "event", "check"])
def test_reordered_keys_with_matching_record_are_rejected(generated, reorder):
    def change(sk):
        if reorder == "top_level":
            ordered = {key: sk[key] for key in sorted(sk)}
            sk.clear()
            sk.update(ordered)
        elif reorder == "event":
            sk["planned_events"][0] = dict(reversed(list(sk["planned_events"][0].items())))
        else:
            sk["checks"][0] = dict(reversed(list(sk["checks"][0].items())))
    errors = _validate_mutated(generated, "CALL_001", change)
    assert any("not the canonical serialization" in e and "CALL_001.json" in e for e in errors), errors


def test_stale_record_hash_is_detected(generated):
    inputs, texts, _ = generated
    texts = dict(texts)
    record = parse_json_text(texts[RECORD])
    record["skeletons"]["CALL_010"]["skeleton_sha256"] = "f" * 64
    texts[RECORD] = serialize_json(record)
    assert any("skeletons differs from the value recomputed" in e for e in validate_skeleton_set(inputs, texts, SCHEMAS))


def test_reordered_record_is_rejected(generated):
    inputs, texts, _ = generated
    texts = dict(texts)
    record = parse_json_text(texts[RECORD])
    texts[RECORD] = serialize_json({key: record[key] for key in sorted(record)})
    assert any("not the canonical serialization" in e and RECORD in e for e in validate_skeleton_set(inputs, texts, SCHEMAS))


# ---------------------------------------------------------------------------------------- record + determinism
def test_record_summarises_the_set(generated):
    record = parse_json_text(generated[1][RECORD])
    assert record["stage"] == "G2B" and list(record["skeletons"]) == CALL_IDS
    s = record["summary"]
    assert (s["skeletons"], s["checks_per_skeleton"], s["edge_case_assignments"]) == (50, 20, 90)
    assert s["planned_events"] == sum(s["event_kinds"].values()) == sum(s["event_statuses"].values())
    assert s["constraints"] == sum(s["constraint_types"].values())
    assert "timestamp" not in json.dumps(record).lower()
    for cid in CALL_IDS:
        assert record["skeletons"][cid]["skeleton_sha256"] == text_hash((RELEASE / skeleton_path(cid)).read_text(encoding="utf-8"))


def test_committed_outputs_validate_and_reproduce():
    assert main(["--dataset-root", str(RELEASE), "--validate"]) == 0
    assert main(["--dataset-root", str(RELEASE), "--check"]) == 0


def test_generation_is_deterministic(generated):
    again = generate(RELEASE, SCHEMAS)[1]
    assert again == generated[1] and len(again) == 51
    assert check_outputs(RELEASE, again) == {"missing": [], "changed": [], "extra": []}


def test_second_cli_generation_is_byte_identical(release):
    before = _snapshot(release)
    assert main(["--dataset-root", str(release), "--overwrite"]) == 0
    assert _snapshot(release) == before


# ---------------------------------------------------------------------------------------- disk safety (mirrors G2A)
def test_existing_outputs_are_not_overwritten_without_flag(release, capsys):
    before = _snapshot(release)
    assert main(["--dataset-root", str(release)]) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert _snapshot(release) == before


def test_fresh_release_generates_once_then_refuses(release):
    _remove_tree(release / SPEC_DIR)
    (release / RECORD).unlink()
    assert main(["--dataset-root", str(release), "--validate"]) == 1
    assert main(["--dataset-root", str(release)]) == 0
    assert main(["--dataset-root", str(release), "--validate"]) == 0
    assert main(["--dataset-root", str(release)]) == 2


def test_check_reports_missing_changed_and_extra(release, capsys):
    (release / skeleton_path("CALL_007")).unlink()
    path = release / skeleton_path("CALL_008")
    path.write_text(path.read_text(encoding="utf-8").replace("EV_001", "EV_001 ", 1), encoding="utf-8", newline="\n")
    (release / SPEC_DIR / "CALL_051.json").write_text("{}", encoding="utf-8")
    assert main(["--dataset-root", str(release), "--check"]) == 1
    err = capsys.readouterr().err
    assert f"missing: {skeleton_path('CALL_007')}" in err and f"changed: {skeleton_path('CALL_008')}" in err
    assert f"extra: {SPEC_DIR}/CALL_051.json" in err
    assert main(["--dataset-root", str(release), "--validate"]) == 1


def test_overwrite_removes_stale_files(release):
    (release / SPEC_DIR / "CALL_999.json").write_text("{}", encoding="utf-8")
    (release / SPEC_DIR / "CALL_001.json.tmp").write_text("partial", encoding="utf-8")
    assert main(["--dataset-root", str(release), "--overwrite"]) == 0
    assert sorted(p.name for p in (release / SPEC_DIR).iterdir()) == [f"{cid}.json" for cid in CALL_IDS]
    assert main(["--dataset-root", str(release), "--validate"]) == 0


def test_record_temp_symlink_is_refused_and_its_target_untouched(release, tmp_path, capsys):
    victim = tmp_path / "victim.txt"
    victim.write_text("PRECIOUS CONTENT", encoding="utf-8")
    link = release / (RECORD + ".tmp")
    _symlink(link, victim)
    before = _snapshot(release)
    for arguments in ([], ["--overwrite"]):
        assert main(["--dataset-root", str(release), *arguments]) == 2
        assert "symlink" in capsys.readouterr().err
    assert victim.read_text(encoding="utf-8") == "PRECIOUS CONTENT" and link.is_symlink()
    assert _snapshot(release) == before


def test_spec_directory_temp_symlink_is_refused(release, tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("PRECIOUS CONTENT", encoding="utf-8")
    link = release / SPEC_DIR / "CALL_001.json.tmp"
    _symlink(link, victim)
    texts = generate(release, SCHEMAS)[1]
    with pytest.raises(ValueError, match="symlink"):
        write_skeletons(release, texts, overwrite=True)
    assert victim.read_text(encoding="utf-8") == "PRECIOUS CONTENT" and link.is_symlink()


def test_expected_output_path_symlink_is_rejected(release, tmp_path):
    path = release / skeleton_path("CALL_003")
    target = tmp_path / "elsewhere.json"
    shutil.copyfile(path, target)
    path.unlink()
    _symlink(path, target)
    original = target.read_bytes()
    inputs = load_inputs(release, SCHEMAS)
    assert any("CALL_003" in e and "not a regular file" in e for e in validate_skeleton_set(inputs, read_disk(release), SCHEMAS))
    assert main(["--dataset-root", str(release), "--validate"]) == 1
    assert main(["--dataset-root", str(release), "--overwrite"]) == 2
    assert target.read_bytes() == original and path.is_symlink()


def test_orphaned_record_temp_is_detected_then_replaced(release, capsys):
    orphan = release / (RECORD + ".tmp")
    orphan.write_text("left behind by an interrupted run", encoding="utf-8")
    assert main(["--dataset-root", str(release), "--validate"]) == 1
    assert "orphaned temporary file" in capsys.readouterr().err
    assert main(["--dataset-root", str(release), "--check"]) == 1
    assert main(["--dataset-root", str(release), "--overwrite"]) == 0
    assert not orphan.exists()
    assert main(["--dataset-root", str(release), "--validate"]) == 0


def test_orphaned_record_temp_blocks_generation_without_overwrite(release):
    _remove_tree(release / SPEC_DIR)
    (release / RECORD).unlink()
    orphan = release / (RECORD + ".tmp")
    orphan.write_text("interrupted", encoding="utf-8")
    assert main(["--dataset-root", str(release)]) == 2
    assert orphan.read_text(encoding="utf-8") == "interrupted" and not (release / SPEC_DIR).exists()


def test_write_refuses_unexpected_subdirectory(release):
    (release / SPEC_DIR / "nested").mkdir()
    texts = generate(release, SCHEMAS)[1]
    with pytest.raises(ValueError, match="subdirectory"):
        write_skeletons(release, texts, overwrite=True)


@pytest.mark.parametrize("reorder", ["top_level", "event"])
def test_validate_enforces_canonical_bytes_on_disk(release, reorder, capsys):
    path = release / skeleton_path("CALL_001")
    original = parse_json_text(path.read_text(encoding="utf-8"))
    sk = parse_json_text(path.read_text(encoding="utf-8"))
    if reorder == "top_level":
        sk = {key: sk[key] for key in sorted(sk)}
    else:
        sk["planned_events"][0] = dict(reversed(list(sk["planned_events"][0].items())))
    path.write_text(serialize_json(sk), encoding="utf-8", newline="\n")
    assert parse_json_text(path.read_text(encoding="utf-8")) == original  # semantically identical
    record_path = release / RECORD
    record = parse_json_text(record_path.read_text(encoding="utf-8"))
    record["skeletons"]["CALL_001"]["skeleton_sha256"] = text_hash(path.read_text(encoding="utf-8"))
    record_path.write_text(serialize_json(record), encoding="utf-8", newline="\n")
    assert main(["--dataset-root", str(release), "--validate"]) == 1
    assert "CALL_001.json: not the canonical serialization" in capsys.readouterr().err
    assert main(["--dataset-root", str(release), "--check"]) == 1


# ---------------------------------------------------------------------------------------- upstream gating
def test_stale_or_mutated_call_spec_is_refused(release, capsys):
    path = release / "authoring" / "call_specs" / "CALL_002.json"
    spec = parse_json_text(path.read_text(encoding="utf-8"))
    entry = next(g for g in spec["gold"] if g["question_id"] == "T04")
    entry["expected"]["expected_value"] += 1
    path.write_text(serialize_json(spec), encoding="utf-8", newline="\n")
    with pytest.raises(G2AStateError, match="G2A outputs are not valid"):
        load_inputs(release, SCHEMAS)
    for arguments in (["--overwrite"], ["--validate"], ["--check"]):
        assert main(["--dataset-root", str(release), *arguments]) == 2
        assert "G2A outputs are not valid" in capsys.readouterr().err


def test_stale_g1_state_is_refused(release, capsys):
    path = release / "authoring" / "call_manifest.csv"
    path.write_bytes(path.read_bytes().replace(b"en-SG", b"en-GB", 1))
    assert main(["--dataset-root", str(release), "--overwrite"]) == 2
    assert "G1 outputs are not valid" in capsys.readouterr().err


def test_stale_g0_input_is_refused_or_detected(release):
    path = release / "edge_case_catalog.json"
    path.write_bytes(path.read_bytes().replace(b"Catalog of deliberate", b"Catalogue of deliberate", 1))
    assert main(["--dataset-root", str(release), "--overwrite"]) == 2


def test_g2b_never_reads_the_g1_csvs_as_inputs():
    package = Path(edge_module.__file__).parent
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    for forbidden in (".csv", "planning_record", "parse_matrix", "parse_manifest", "parse_assignments", "load_matrix", "g1.MATRIX", "g1.MANIFEST", "g1.ASSIGNMENTS"):
        assert forbidden not in source, forbidden


def test_package_has_no_llm_or_network_imports():
    package = Path(edge_module.__file__).parent
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    for forbidden in ("anthropic", "openai", "boto3", "requests", "httpx", "urllib.request", "socket", "google.genai", "polly", "ssml"):
        assert f"import {forbidden}" not in source and f"from {forbidden}" not in source
    assert "<speak" not in source and "SynthesizeSpeech" not in source


def test_source_hashes_match_current_g0_files(inputs):
    assert inputs.g0_hashes == g0_hashes(RELEASE)


def test_order_replacement_constraints_chain_initial_invalidation_final(inputs, skeletons):
    """Every order-replacement call keeps initial ORDER_STATEMENT -> ORDER_INVALIDATION -> final ORDER_STATEMENT."""
    calls = [c for c in CALL_IDS if any(e["edge_case_id"] == "EC_COMPLETE_ORDER_REPLACEMENT" for e in inputs.specs[c]["edge_cases"])]
    assert calls
    for call_id in calls:
        sk = skeletons[call_id]
        aid = next(e["assignment_id"] for e in inputs.specs[call_id]["edge_cases"] if e["edge_case_id"] == "EC_COMPLETE_ORDER_REPLACEMENT")
        initial, invalidation, final = evs(sk, aid)
        assert (initial["kind"], invalidation["kind"], final["kind"]) == ("ORDER_STATEMENT", "ORDER_INVALIDATION", "ORDER_STATEMENT")
        source = {"type": "EDGE_ASSIGNMENT", "id": aid}
        orders = [(c["before"], c["after"]) for c in sk["constraints"] if c["type"] == "ORDER" and c["source"] == source]
        assert (initial["event_id"], invalidation["event_id"]) in orders
        assert (invalidation["event_id"], final["event_id"]) in orders
        assert (final["event_id"], initial["event_id"]) not in orders and (invalidation["event_id"], initial["event_id"]) not in orders


def test_constraints_with_same_endpoints_but_different_provenance_are_both_kept():
    """Deduplication is exact: type, endpoints, parameters and provenance must all match."""
    edge = {"type": "EDGE_ASSIGNMENT", "id": "EA_001"}
    rule = {"type": "ORDERING_RULE", "id": "C03_AFTER_C02"}
    plan = model.Plan(None)
    plan.order("EV_001", "EV_002", edge)
    plan.order("EV_001", "EV_002", rule)
    plan.order("EV_001", "EV_002", edge)  # exact duplicate: merged
    plan.same_turn("EV_002", "EV_001", edge)
    plan.same_turn("EV_001", "EV_002", edge)  # same pair in the other argument order: merged
    plan.same_turn("EV_001", "EV_002", rule)
    assert [(c["type"], c["source"]["id"]) for c in plan.constraints] == [
        ("ORDER", "EA_001"), ("ORDER", "C03_AFTER_C02"), ("SAME_TURN", "EA_001"), ("SAME_TURN", "C03_AFTER_C02")]
