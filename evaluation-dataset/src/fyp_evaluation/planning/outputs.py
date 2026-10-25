"""G1 output contracts: canonical serialization, strict loaders, record and safe writes.

Text outputs are UTF-8 with LF newlines. Hashes are SHA-256 over bytes with
CRLF normalised to LF, so a Windows `core.autocrlf` checkout hashes the same.
"""

import csv
import hashlib
import io
import json
import os
import re
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator

from fyp_evaluation.authoring.coverage import load_matrix
from fyp_evaluation.validation.schemas import load_json
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR, is_multilingual, load_stage0, stage0_files

from .allocate import build_plan
from .validate import typed_gold, validate_plan

GENERATOR_VERSION = "1.1.0"
MULTILINGUAL_GENERATOR_VERSION = "2.0.0"
RECORD_SCHEMA_VERSION = "1.0"
MULTILINGUAL_RECORD_SCHEMA_VERSION = "2.0"
MANIFEST = "authoring/call_manifest.csv"
MATRIX = "authoring/coverage_matrix.csv"
ASSIGNMENTS = "authoring/edge_case_assignments.csv"
RECORD = "authoring/planning_record.json"
OUTPUTS = (MANIFEST, MATRIX, ASSIGNMENTS, RECORD)
MANIFEST_HEADER = [
    "call_id", "call_intents", "language", "duration_profile",
    "target_duration_min_seconds", "target_duration_max_seconds",
]
ASSIGNMENT_HEADER = [
    "assignment_id", "call_id", "edge_case_id", "family", "target_question_ids", "parameters_json",
]
# Multilingual (generation plan 2.0) releases add the frozen language profile, mixing style and edge-case language variant.
MANIFEST_HEADER_V2 = [
    "call_id", "call_intents", "language_profile", "mixing_style", "duration_profile",
    "target_duration_min_seconds", "target_duration_max_seconds",
]
ASSIGNMENT_HEADER_V2 = [
    "assignment_id", "call_id", "edge_case_id", "family", "target_question_ids", "language_variant", "parameters_json",
]
RNG_DESCRIPTION = (
    "Python random.Random (MT19937) per named decision stream; stream seed = first 8 bytes "
    "(big-endian) of SHA-256('<random_seed>|<attempt>|<stream name>')."
)


def sha256(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _csv_text(header, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _cell(value):
    return value if isinstance(value, str) else json.dumps(value)


def serialize_plan(documents, result):
    """Return {relative path: text} for the three tabular outputs."""
    if is_multilingual(documents):
        return _serialize_multilingual(documents, result)
    checklist = documents["checklist"]
    calls = [c["call_id"] for c in result["calls"]]
    return {
        MANIFEST: _csv_text(MANIFEST_HEADER, [
            [c["call_id"], "|".join(c["call_intents"]), c["language"], c["duration_profile"],
             c["target_duration_min_seconds"], c["target_duration_max_seconds"]]
            for c in result["calls"]
        ]),
        MATRIX: _csv_text(["question_id", *calls], [
            [item["id"], *(_cell(result["gold"][c][item["id"]]) for c in calls)] for item in checklist
        ]),
        ASSIGNMENTS: _csv_text(ASSIGNMENT_HEADER, [
            [a["assignment_id"], a["call_id"], a["edge_case_id"], a["family"],
             "|".join(a["target_question_ids"]), canonical_json(a["parameters"])]
            for a in result["assignments"]
        ]),
    }


def _serialize_multilingual(documents, result):
    checklist = documents["checklist"]
    calls = [c["call_id"] for c in result["calls"]]
    return {
        MANIFEST: _csv_text(MANIFEST_HEADER_V2, [
            [c["call_id"], "|".join(c["call_intents"]), c["language_profile"], c["mixing_style"], c["duration_profile"],
             c["target_duration_min_seconds"], c["target_duration_max_seconds"]]
            for c in result["calls"]
        ]),
        MATRIX: _csv_text(["question_id", *calls], [
            [item["id"], *(_cell(result["gold"][c][item["id"]]) for c in calls)] for item in checklist
        ]),
        ASSIGNMENTS: _csv_text(ASSIGNMENT_HEADER_V2, [
            [a["assignment_id"], a["call_id"], a["edge_case_id"], a["family"], "|".join(a["target_question_ids"]),
             a["language_variant"], canonical_json(a["parameters"])]
            for a in result["assignments"]
        ]),
    }


def _header(text):
    return next(csv.reader(io.StringIO(text.replace("\r\n", "\n"), newline="")), [])


def _rows(text, header, label):
    try:
        rows = list(csv.reader(io.StringIO(text.replace("\r\n", "\n"), newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"{label}: malformed CSV: {error}") from error
    if not rows or rows[0] != header:
        raise ValueError(f"{label}: header must be exactly {header}")
    for line, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError(f"{label}: row {line} has {len(row)} cells, expected {len(header)}")
        if any(not cell or cell != cell.strip() for cell in row):
            raise ValueError(f"{label}: row {line} has a blank or padded cell")
    return rows[1:]


def _integer(text, label):
    if not re.fullmatch(r"[1-9][0-9]*", text):
        raise ValueError(f"{label}: {text!r} is not a canonical positive integer")
    return int(text)


def parse_manifest(text):
    """Parse call_manifest.csv; the header selects the 1.0 layout or the multilingual 2.0 layout."""
    if _header(text) == MANIFEST_HEADER_V2:
        return _parse_manifest_v2(text)
    rows = []
    for row in _rows(text, MANIFEST_HEADER, "call_manifest"):
        call_id, intents, language, profile, low, high = row
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*(\|[A-Z][A-Z0-9_]*)*", intents):
            raise ValueError(f"call_manifest: {call_id}: call_intents must be pipe-delimited symbols")
        rows.append({
            "call_id": call_id, "call_intents": intents.split("|"), "language": language,
            "duration_profile": profile,
            "target_duration_min_seconds": _integer(low, f"call_manifest: {call_id}"),
            "target_duration_max_seconds": _integer(high, f"call_manifest: {call_id}"),
        })
    return rows


def _parse_manifest_v2(text):
    rows = []
    for row in _rows(text, MANIFEST_HEADER_V2, "call_manifest"):
        call_id, intents, language_profile, mixing_style, profile, low, high = row
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*(\|[A-Z][A-Z0-9_]*)*", intents):
            raise ValueError(f"call_manifest: {call_id}: call_intents must be pipe-delimited symbols")
        rows.append({
            "call_id": call_id, "call_intents": intents.split("|"), "language_profile": language_profile,
            "mixing_style": mixing_style, "duration_profile": profile,
            "target_duration_min_seconds": _integer(low, f"call_manifest: {call_id}"),
            "target_duration_max_seconds": _integer(high, f"call_manifest: {call_id}"),
        })
    return rows


def _reject_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON key")
    return dict(pairs)


def parse_assignments(text):
    """Parse edge_case_assignments.csv; the header selects the 1.0 layout or the multilingual 2.0 layout."""
    multilingual = _header(text) == ASSIGNMENT_HEADER_V2
    rows = []
    for row in _rows(text, ASSIGNMENT_HEADER_V2 if multilingual else ASSIGNMENT_HEADER, "edge_case_assignments"):
        if multilingual:
            assignment_id, call_id, case_id, family, targets, variant, parameters = row
        else:
            assignment_id, call_id, case_id, family, targets, parameters = row
        try:
            params = json.loads(parameters, object_pairs_hook=_reject_duplicates)
        except ValueError as error:
            raise ValueError(f"edge_case_assignments: {assignment_id}: invalid parameters_json: {error}") from error
        if not isinstance(params, dict) or canonical_json(params) != parameters:
            raise ValueError(f"edge_case_assignments: {assignment_id}: parameters_json must be canonical JSON")
        entry = {
            "assignment_id": assignment_id, "call_id": call_id, "edge_case_id": case_id, "family": family,
            "target_question_ids": targets.split("|"),
        }
        if multilingual:
            entry["language_variant"] = variant
        entry["parameters"] = params
        rows.append(entry)
    return rows


def parse_matrix(text):
    """Parse in-memory matrix text with the same structural rules as `load_matrix`.

    Cell typing is left to `check_matrix`/`compile_expected` in `typed_gold`.
    """
    try:
        lines = list(csv.reader(io.StringIO(text.replace("\r\n", "\n"), newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"coverage_matrix: malformed CSV: {error}") from error
    if not lines or not lines[0] or lines[0][0] != "question_id":
        raise ValueError("coverage_matrix: first header must be question_id")
    header = lines[0]
    if len(header) < 2 or len(header) != len(set(header)):
        raise ValueError("coverage_matrix: headers must include unique call columns")
    matrix, seen = {call_id: {} for call_id in header[1:]}, set()
    for row in lines[1:]:
        if len(row) != len(header):
            raise ValueError("coverage_matrix: ragged row")
        if row[0] in seen:
            raise ValueError(f"coverage_matrix: duplicate question ID {row[0]}")
        seen.add(row[0])
        for call_id, cell in zip(header[1:], row[1:]):
            matrix[call_id][row[0]] = cell
    return matrix


def summarize(documents, manifest, gold, assignments):
    """Deterministic review counts recorded in planning_record.json."""
    if is_multilingual(documents):
        return summarize_multilingual(documents, manifest, gold, assignments)
    checklist = documents["checklist"]
    calls = [r["call_id"] for r in manifest]
    compliance = [i["id"] for i in checklist if i["type"] == "COMPLIANCE"]
    field = {i["field"]: i["id"] for i in checklist if i["type"] != "COMPLIANCE"}
    cases = [c["id"] for c in documents["edge_case_catalog"]["edge_cases"]]
    families = list(documents["generation_plan"]["edge_case_distribution"]["family_quotas"])
    per_call = Counter(a["call_id"] for a in assignments)
    fails = {c: sum(gold[c][q] == "FAIL" for q in compliance) for c in calls}
    composition = Counter(
        ("all_pass" if not fails[c] else "some_fail") + ("_with_edge_cases" if per_call[c] else "_without_edge_cases")
        for c in calls
    )

    def ordered(counter, keys):
        return {key: counter[key] for key in keys}

    targets = Counter(q for a in assignments for q in a["target_question_ids"])
    return {
        "calls": len(calls),
        "duration_profiles": ordered(Counter(r["duration_profile"] for r in manifest),
                                     [d["profile"] for d in documents["generation_plan"]["duration_profiles"]]),
        "call_composition": ordered(composition, list(documents["generation_plan"]["call_composition_counts"])),
        "compliance_verdicts": {
            q: {"PASS": sum(gold[c][q] == "PASS" for c in calls), "FAIL": sum(gold[c][q] == "FAIL" for c in calls)}
            for q in compliance
        },
        "fails_per_call": {str(n): count for n, count in sorted(Counter(fails.values()).items())},
        "total_compliance_fail_pairs": sum(fails.values()),
        "total_edge_case_assignments": len(assignments),
        "transaction_direction": ordered(Counter(gold[c][field["transaction_direction"]] for c in calls),
                                         list(documents["generation_plan"]["extraction_distribution"]["transaction_direction_counts"])),
        "order_type": ordered(Counter(gold[c][field["order_type"]] for c in calls),
                              list(documents["generation_plan"]["extraction_distribution"]["order_type_counts"])),
        "transaction_currency": ordered(Counter(gold[c][field["transaction_currency"]] for c in calls),
                                        list(documents["generation_plan"]["extraction_distribution"]["transaction_currency_counts"])),
        "transaction_instrument": ordered(
            Counter(gold[c][field["transaction_instrument"]] for c in calls),
            [p["name"] for p in documents["generation_plan"]["extraction_distribution"]["product_pool"]]),
        "edge_case_families": ordered(Counter(a["family"] for a in assignments), families),
        "edge_case_ids": ordered(Counter(a["edge_case_id"] for a in assignments), cases),
        "edge_case_target_questions": {i["id"]: targets[i["id"]] for i in checklist if targets[i["id"]]},
        "edge_cases_per_call": {str(n): count for n, count in sorted(Counter(per_call[c] for c in calls).items())},
    }


def _profile_summary(documents, p, rows, gold, assignments):
    """Review counts for one language profile of a multilingual release (keys in configuration order)."""
    checklist = documents["checklist"]
    extraction = documents["generation_plan"]["extraction_distribution"]
    calls = [r["call_id"] for r in rows]
    compliance = [i["id"] for i in checklist if i["type"] == "COMPLIANCE"]
    field = {i["field"]: i["id"] for i in checklist if i["type"] != "COMPLIANCE"}
    catalog = documents["edge_case_catalog"]["edge_cases"]
    case_ids = [c["id"] for c in catalog]
    families = list(dict.fromkeys(c["family"] for c in catalog))
    per_call = Counter(a["call_id"] for a in assignments)
    fails = {c: sum(gold[c][q] == "FAIL" for q in compliance) for c in calls}
    duration = {r["call_id"]: r["duration_profile"] for r in rows}
    composition = Counter(
        ("all_pass" if not fails[c] else "some_fail") + ("_with_edge_cases" if per_call[c] else "_without_edge_cases")
        for c in calls
    )

    def ordered(counter, keys):
        return {key: counter[key] for key in keys}

    targets = Counter(q for a in assignments for q in a["target_question_ids"])
    loads = {}
    for c in calls:
        loads.setdefault(duration[c], Counter())[fails[c] + per_call[c]] += 1
    variants = {}
    for a in assignments:
        if a["language_variant"] != "NONE":
            variants.setdefault(a["edge_case_id"], Counter())[a["language_variant"]] += 1
    pair = Counter((gold[c]["C02"], gold[c]["C03"]) for c in calls) if {"C02", "C03"} <= set(compliance) else Counter()
    return {
        "calls": len(calls),
        "duration_profiles": ordered(Counter(duration.values()), list(p["duration_counts"])),
        "call_composition": ordered(composition, list(p["call_composition_counts"])),
        "compliance_verdicts": {
            q: {"PASS": sum(gold[c][q] == "PASS" for c in calls), "FAIL": sum(gold[c][q] == "FAIL" for c in calls)}
            for q in compliance
        },
        "c02_c03_outcomes": {f"{a}/{b}": pair[(a, b)] for a, b in (("PASS", "PASS"), ("PASS", "FAIL"), ("FAIL", "FAIL"), ("FAIL", "PASS"))},
        "fails_per_call": {str(n): count for n, count in sorted(Counter(fails.values()).items())},
        "total_compliance_fail_pairs": sum(fails.values()),
        "total_edge_case_assignments": len(assignments),
        "transaction_direction": ordered(Counter(gold[c][field["transaction_direction"]] for c in calls), extraction["transaction_directions"]),
        "order_type": ordered(Counter(gold[c][field["order_type"]] for c in calls), extraction["order_types"]),
        "transaction_currency": ordered(Counter(gold[c][field["transaction_currency"]] for c in calls), extraction["transaction_currencies"]),
        "transaction_instrument": ordered(Counter(gold[c][field["transaction_instrument"]] for c in calls),
                                          [product["name"] for product in extraction["product_pool"]]),
        "edge_case_families": ordered(Counter(a["family"] for a in assignments), families),
        "edge_case_ids": ordered(Counter(a["edge_case_id"] for a in assignments), case_ids),
        "edge_case_target_questions": {i["id"]: targets[i["id"]] for i in checklist if targets[i["id"]]},
        "edge_cases_per_call": {str(n): count for n, count in sorted(Counter(per_call[c] for c in calls).items())},
        "language_variants": {case_id: dict(sorted(variants[case_id].items())) for case_id in case_ids if case_id in variants},
        "mixing_styles": dict(sorted(Counter(r["mixing_style"] for r in rows).items())),
        "target_exposure_calls": {q: len({a["call_id"] for a in assignments if q in a["target_question_ids"]})
                                  for q in p["target_exposure_caps"]},
        "fails_plus_edge_cases_by_duration": {d: {str(n): loads[d][n] for n in sorted(loads[d])} for d in p["duration_counts"] if d in loads},
    }


def summarize_multilingual(documents, manifest, gold, assignments):
    by_profile = {}
    for p in documents["generation_plan"]["language_profile_plans"]:
        name = p["language_profile"]
        rows = [r for r in manifest if r["language_profile"] == name]
        members = {r["call_id"] for r in rows}
        by_profile[name] = _profile_summary(documents, p, rows, gold, [a for a in assignments if a["call_id"] in members])
    compliance = [i["id"] for i in documents["checklist"] if i["type"] == "COMPLIANCE"]
    return {
        "calls": len(manifest),
        "language_profiles": {name: block["calls"] for name, block in by_profile.items()},
        "total_compliance_fail_pairs": sum(block["total_compliance_fail_pairs"] for block in by_profile.values()),
        "total_edge_case_assignments": len(assignments),
        "total_language_variants": sum(a["language_variant"] != "NONE" for a in assignments),
        "compliance_fail_counts": {q: sum(block["compliance_verdicts"][q]["FAIL"] for block in by_profile.values()) for q in compliance},
        "by_language_profile": by_profile,
    }


def build_record(documents, dataset_root, texts, result, summary):
    root = Path(dataset_root)
    multilingual = is_multilingual(documents)
    return {
        "schema_version": MULTILINGUAL_RECORD_SCHEMA_VERSION if multilingual else RECORD_SCHEMA_VERSION,
        "stage": "G1",
        "dataset_id": documents["normalization"]["dataset_id"],
        "generator": {"package": "fyp_evaluation.planning",
                      "version": MULTILINGUAL_GENERATOR_VERSION if multilingual else GENERATOR_VERSION, "rng": RNG_DESCRIPTION},
        "random_seed": documents["generation_plan"]["random_seed"],
        "attempts": {"accepted_attempt": result["attempt"], "rejected_attempts": result["rejected_attempts"]},
        "hashing": "SHA-256 over UTF-8 bytes with CRLF normalised to LF",
        "inputs": {relative: sha256((root / relative).read_bytes()) for relative in stage0_files(documents).values()},
        "outputs": {relative: sha256(texts[relative].encode("utf-8")) for relative in (MANIFEST, MATRIX, ASSIGNMENTS)},
        "summary": summary,
    }


def _record_text(record):
    return json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def generate(dataset_root, schema_dir=DEFAULT_SCHEMA_DIR):
    """Plan, serialize and fully validate in memory; return ({path: text}, summary). Writes nothing."""
    documents = load_stage0(dataset_root)
    result = build_plan(documents)
    texts = serialize_plan(documents, result)
    manifest = parse_manifest(texts[MANIFEST])
    matrix = parse_matrix(texts[MATRIX])
    assignments = parse_assignments(texts[ASSIGNMENTS])
    if errors := validate_plan(documents, manifest, matrix, assignments, schema_dir):
        raise ValueError("Generated plan failed validation (generator defect):\n" + "\n".join(errors))
    gold, _ = typed_gold(documents, matrix, [r["call_id"] for r in manifest], schema_dir)
    summary = summarize(documents, manifest, gold, assignments)
    record = build_record(documents, dataset_root, texts, result, summary)
    if errors := validate_record_shape(record, schema_dir):
        raise ValueError("Planning record failed its schema:\n" + "\n".join(errors))
    texts[RECORD] = _record_text(record)
    return texts, summary


def validate_record_shape(record, schema_dir=DEFAULT_SCHEMA_DIR):
    schema = load_json(Path(schema_dir) / "stage1" / "planning_record.schema.json")
    Draft202012Validator.check_schema(schema)
    return [f"planning_record: {e.json_path}: {e.message}" for e in Draft202012Validator(schema).iter_errors(record)]


def write_outputs(dataset_root, texts, *, overwrite=False):
    """Write validated texts; refuse existing files unless overwrite. Each file is replaced atomically."""
    root = Path(dataset_root)
    targets = {relative: root / relative for relative in OUTPUTS}
    for path in targets.values():
        if path.is_symlink():
            raise ValueError(f"Refusing symlink output: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}; use --overwrite explicitly")
    for relative, path in targets.items():
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(texts[relative])
        os.replace(temporary, path)
    return list(targets.values())


def check_outputs(dataset_root, texts):
    """Return the output paths whose on-disk content differs from the regenerated texts."""
    root = Path(dataset_root)
    return [
        relative for relative in OUTPUTS
        if not (root / relative).is_file()
        or (root / relative).read_bytes().replace(b"\r\n", b"\n") != texts[relative].encode("utf-8")
    ]


def validate_written(dataset_root, schema_dir=DEFAULT_SCHEMA_DIR):
    """Validate the four files on disk, through the existing matrix loader and the record hashes."""
    root = Path(dataset_root)
    documents = load_stage0(root)
    errors = []
    try:
        manifest = parse_manifest((root / MANIFEST).read_text(encoding="utf-8"))
        assignments = parse_assignments((root / ASSIGNMENTS).read_text(encoding="utf-8"))
        call_ids = [r["call_id"] for r in manifest]
        matrix = load_matrix(root / MATRIX, checklist=documents["checklist"], call_ids=call_ids)
        record = load_json(root / RECORD)
    except (OSError, ValueError) as error:
        return [str(error)]
    errors += validate_plan(documents, manifest, matrix, assignments, schema_dir)
    errors += validate_record_shape(record, schema_dir)
    if errors:
        return errors
    if set(record["inputs"]) != set(stage0_files(documents).values()):
        errors.append("planning_record: inputs must hash exactly the release's G0 files")
    for relative in stage0_files(documents).values():
        if record["inputs"].get(relative) != sha256((root / relative).read_bytes()):
            errors.append(f"planning_record: input hash differs for {relative}")
    for relative in (MANIFEST, MATRIX, ASSIGNMENTS):
        if record["outputs"].get(relative) != sha256((root / relative).read_bytes()):
            errors.append(f"planning_record: output hash differs for {relative}")
    gold, _ = typed_gold(documents, matrix, call_ids, schema_dir)
    if record["summary"] != summarize(documents, manifest, gold, assignments):
        errors.append("planning_record: summary differs from the written outputs")
    if record["random_seed"] != documents["generation_plan"]["random_seed"]:
        errors.append("planning_record: random_seed differs from generation_plan.json")
    return errors
