"""Independent G2A validation of a call-spec set against the current G1 sources.

Works on {relative path: text} so the same checks run on freshly compiled
in-memory texts and on the files actually on disk. Gold, metadata and edge
cases are rebuilt from the G1 files (never from the record) and compared with
strict JSON types, so 1200000 never matches 1200000.0. The decisive check is
the round trip: the specs must project back to byte-identical G1 CSVs.
"""

from pathlib import Path

from jsonschema import Draft202012Validator

from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.validation.schemas import load_json
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .compile import (
    RECORD, build_record, build_specs, gold_entries, parse_json_text, project_to_g1_texts, serialize_json, spec_path,
)

SCHEMA_NAMES = ("call_spec", "call_spec_record")


def _validator(schema_dir, name):
    schema = load_json(Path(schema_dir) / "stage2" / f"{name}.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def check_call_spec_schemas(schema_dir=DEFAULT_SCHEMA_DIR):
    for name in SCHEMA_NAMES:
        _validator(schema_dir, name)


def _floats(value, path="$"):
    if isinstance(value, float):
        yield path
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _floats(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _floats(child, f"{path}[{index}]")


def _same(left, right):
    """Equality that also distinguishes JSON types (1 vs 1.0 vs true) and key order does not matter."""
    return g1.canonical_json(left) == g1.canonical_json(right)


def _edge_entry(assignment):
    return {key: assignment[key] for key in (
        "assignment_id", "edge_case_id", "family", "target_question_ids", "parameters")}


def _spec_errors(spec, call_id, src, validator):
    errors = [f"{e.json_path}: {e.message}" for e in validator.iter_errors(spec)]
    if errors:
        return errors
    if spec["call_id"] != call_id:
        errors.append(f"call_id {spec['call_id']} does not match the file name")
    if spec["dataset_id"] != src.dataset_id:
        errors.append(f"dataset_id {spec['dataset_id']} differs from dataset_manifest.json")

    row = next(r for r in src.manifest if r["call_id"] == call_id)
    meta = spec["metadata"]
    manifest_meta = {key: row[key] for key in (
        "call_intents", "language", "duration_profile", "target_duration_min_seconds", "target_duration_max_seconds")}
    if not _same(meta, manifest_meta):
        errors.append("metadata differs from call_manifest.csv")
    vocabulary = src.documents["generation_plan"]["call_intent_vocabulary"]
    intents = meta["call_intents"]
    if any(i not in vocabulary for i in intents) or intents != sorted(intents, key=vocabulary.index):
        errors.append(f"call_intents {intents} must be unique vocabulary terms in vocabulary order")
    if meta["target_duration_min_seconds"] > meta["target_duration_max_seconds"]:
        errors.append("target_duration_min_seconds exceeds target_duration_max_seconds")

    checklist_ids = [item["id"] for item in src.documents["checklist"]]
    if [g["question_id"] for g in spec["gold"]] != checklist_ids:
        errors.append(f"gold must list exactly the {len(checklist_ids)} checklist questions in checklist order")
    else:
        for entry, wanted in zip(spec["gold"], gold_entries(src, call_id)):
            if not _same(entry, wanted):
                errors.append(f"gold {entry['question_id']} differs from coverage_matrix.csv (value or JSON type)")
    errors += [f"floating-point number at {path}; numbers are JSON integers in this release" for path in _floats(spec)]

    wanted_edges = [_edge_entry(a) for a in src.assignments if a["call_id"] == call_id]
    if len(spec["edge_cases"]) != len(wanted_edges):
        errors.append(f"has {len(spec['edge_cases'])} edge cases; edge_case_assignments.csv assigns {len(wanted_edges)}")
    else:
        for entry, wanted in zip(spec["edge_cases"], wanted_edges):
            if not _same(entry, wanted):
                errors.append(f"edge case {entry['assignment_id']} differs from edge_case_assignments.csv")
    if not _same(spec["source"], src.hashes):
        errors.append("source hashes differ from the current G1 files")
    return errors


def validate_spec_set(src, texts, schema_dir=DEFAULT_SCHEMA_DIR):
    """Return every violation found in a set of spec texts ([] when valid)."""
    spec_validator = _validator(schema_dir, "call_spec")
    record_validator = _validator(schema_dir, "call_spec_record")
    expected = {spec_path(call_id): call_id for call_id in src.call_ids}
    try:
        canonical_specs = build_specs(src)
    except ValueError as error:
        return [f"cannot rebuild the canonical call specs from the G1 files: {error}"]
    canonical_texts = {spec["call_id"]: serialize_json(spec) for spec in canonical_specs}
    errors = []
    for rel in sorted(texts):
        if rel == RECORD or rel in expected:
            continue
        hint = "; orphaned temporary file from an interrupted run, rerun with --overwrite" if rel.endswith(".tmp") else ""
        errors.append(f"unexpected file {rel}{hint}")
    errors += [f"missing {rel}" for rel in expected if rel not in texts]

    specs = {}
    for rel, call_id in expected.items():
        if rel not in texts:
            continue
        if texts[rel] is None:
            errors.append(f"{rel}: unreadable or not a regular file")
            continue
        try:
            spec = parse_json_text(texts[rel])
        except ValueError as error:
            errors.append(f"{rel}: invalid JSON: {error}")
            continue
        problems = _spec_errors(spec, call_id, src, spec_validator)
        if problems:
            errors += [f"{rel}: {problem}" for problem in problems]
        elif texts[rel] != canonical_texts[call_id]:
            errors.append(f"{rel}: not the canonical serialization rebuilt from the frozen G1 inputs (key order or formatting)")
        else:
            specs[call_id] = spec

    complete = len(specs) == len(expected)
    if complete:
        ordered = [specs[call_id] for call_id in src.call_ids]
        used = sorted(e["assignment_id"] for spec in ordered for e in spec["edge_cases"])
        if used != sorted(a["assignment_id"] for a in src.assignments):
            errors.append("edge-case assignments must each appear exactly once, in their owning call")
        projected = project_to_g1_texts(ordered, src.documents)
        planning = load_json(src.root / g1.RECORD)
        for rel in (g1.MANIFEST, g1.MATRIX, g1.ASSIGNMENTS):
            frozen = (src.root / rel).read_bytes().decode("utf-8").replace("\r\n", "\n")
            if projected[rel] != frozen:
                errors.append(f"projecting the call specs back does not reproduce {rel} byte-for-byte")
            if planning["outputs"].get(rel) != g1.sha256(projected[rel].encode("utf-8")):
                errors.append(f"reconstructed {rel} does not match its hash in planning_record.json")

    if RECORD not in texts:
        errors.append(f"missing {RECORD}")
    elif texts[RECORD] is None:
        errors.append(f"{RECORD}: unreadable or not a regular file")
    else:
        try:
            record = parse_json_text(texts[RECORD])
        except ValueError as error:
            errors.append(f"{RECORD}: invalid JSON: {error}")
        else:
            shape = [f"{e.json_path}: {e.message}" for e in record_validator.iter_errors(record)]
            errors += [f"{RECORD}: {problem}" for problem in shape]
            if not shape:
                rebuilt = build_record(src.dataset_id, src.hashes, canonical_texts, canonical_specs)
                for key in rebuilt:
                    if not _same(record[key], rebuilt[key]):
                        errors.append(f"{RECORD}: {key} differs from the value recomputed from the frozen G1 inputs")
                if texts[RECORD] != serialize_json(rebuilt):
                    errors.append(f"{RECORD}: not the canonical serialization rebuilt from the frozen G1 inputs")
    return errors
