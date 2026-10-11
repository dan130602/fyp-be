"""Gate on the frozen G0/G1 state, then compile and project call specs.

Compilation is a pure join. Gold cells are typed with the existing
`compile_expected` through the G1 adapter, so no new value or representation
is introduced; projection back to the three G1 CSVs reuses G1's own serializer.
"""

import json
from copy import deepcopy
from pathlib import Path

from fyp_evaluation.authoring.coverage import compile_expected
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.planning.adapter import legacy_normalization
from fyp_evaluation.validation.schemas import load_json
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR, load_stage0, validate_stage0_release

SCHEMA_VERSION = "1.0"
STAGE = "G2A"
SPEC_DIR = "authoring/call_specs"
RECORD = "authoring/call_spec_record.json"
CHECKLIST = "checklist.json"
G1_SOURCES = {
    "planning_record_sha256": g1.RECORD,
    "call_manifest_sha256": g1.MANIFEST,
    "coverage_matrix_sha256": g1.MATRIX,
    "edge_case_assignments_sha256": g1.ASSIGNMENTS,
}


class G1StateError(ValueError):
    """The G0/G1 state on disk is not valid, so nothing may be compiled from it."""


def spec_path(call_id):
    return f"{SPEC_DIR}/{call_id}.json"


class G1Sources:
    """The frozen G0 documents and the three parsed G1 files, with their hashes."""

    def __init__(self, root, documents, manifest, matrix, assignments, hashes, dataset_id):
        self.root, self.documents, self.manifest = Path(root), documents, manifest
        self.matrix, self.assignments, self.hashes, self.dataset_id = matrix, assignments, hashes, dataset_id
        self.call_ids = [row["call_id"] for row in manifest]
        self.normalization = legacy_normalization(
            documents["checklist"], documents["check_semantics"], documents["normalization"])


def read_sources(root):
    """Parse the G1 files and hash them, with no validation (callers gate first)."""
    root = Path(root)
    documents = load_stage0(root)
    texts = {rel: (root / rel).read_text(encoding="utf-8") for rel in (g1.MANIFEST, g1.MATRIX, g1.ASSIGNMENTS)}
    hashes = {key: g1.sha256((root / rel).read_bytes()) for key, rel in G1_SOURCES.items()}
    hashes["checklist_sha256"] = g1.sha256((root / CHECKLIST).read_bytes())
    return G1Sources(
        root, documents, g1.parse_manifest(texts[g1.MANIFEST]), g1.parse_matrix(texts[g1.MATRIX]),
        g1.parse_assignments(texts[g1.ASSIGNMENTS]), hashes, load_json(root / "dataset_manifest.json")["dataset_id"],
    )


def load_g1_sources(root, schema_dir=DEFAULT_SCHEMA_DIR):
    """Refuse unless G0 and G1 validate on disk (never trust planning_record.json alone)."""
    if errors := validate_stage0_release(root, schema_dir):
        raise G1StateError("G0 contracts are not valid:\n" + "\n".join(errors))
    if errors := g1.validate_written(root, schema_dir):
        raise G1StateError("G1 outputs are not valid; regenerate or fix G1 first:\n" + "\n".join(errors))
    return read_sources(root)


def gold_entries(src, call_id):
    """Typed gold for one call, in checklist order, from the raw coverage-matrix cells."""
    entries = []
    for item in src.documents["checklist"]:
        expected = compile_expected(item, src.normalization, src.matrix[call_id][item["id"]])
        if expected.get("expected_value", "") is None:
            raise ValueError(f"{call_id}/{item['id']}: a null gold value is not allowed in this release")
        entries.append({"question_id": item["id"], "expected": expected})
    return entries


def build_specs(src):
    """One spec per manifest call, in manifest order; assignments keep their G1 file order."""
    by_call = {}
    for assignment in src.assignments:
        by_call.setdefault(assignment["call_id"], []).append(assignment)
    unknown = sorted(set(by_call) - set(src.call_ids))
    if unknown:
        raise ValueError(f"Edge-case assignments reference calls missing from the manifest: {unknown}")
    specs = []
    for row in src.manifest:
        call_id = row["call_id"]
        specs.append({
            "schema_version": SCHEMA_VERSION,
            "stage": STAGE,
            "dataset_id": src.dataset_id,
            "call_id": call_id,
            "metadata": {
                "call_intents": list(row["call_intents"]),
                "language": row["language"],
                "duration_profile": row["duration_profile"],
                "target_duration_min_seconds": row["target_duration_min_seconds"],
                "target_duration_max_seconds": row["target_duration_max_seconds"],
            },
            "gold": gold_entries(src, call_id),
            "edge_cases": [
                {
                    "assignment_id": a["assignment_id"],
                    "edge_case_id": a["edge_case_id"],
                    "family": a["family"],
                    "target_question_ids": list(a["target_question_ids"]),
                    "parameters": deepcopy(a["parameters"]),
                }
                for a in by_call.get(call_id, [])
            ],
            "source": dict(src.hashes),
        })
    return specs


def project_to_g1_texts(specs, documents):
    """Rebuild the three G1 CSV texts from specs using G1's own canonical serializer."""
    calls, gold, assignments = [], {}, []
    for spec in specs:
        meta, call_id = spec["metadata"], spec["call_id"]
        calls.append({"call_id": call_id, **{key: meta[key] for key in (
            "call_intents", "language", "duration_profile",
            "target_duration_min_seconds", "target_duration_max_seconds")}})
        gold[call_id] = {}
        for entry in spec["gold"]:
            expected = entry["expected"]
            gold[call_id][entry["question_id"]] = (
                expected["expected_verdict"] if expected["kind"] == "compliance" else expected["expected_value"])
        for edge in spec["edge_cases"]:
            assignments.append({"call_id": call_id, **edge})
    return g1.serialize_plan(documents, {"calls": calls, "gold": gold, "assignments": assignments})


GENERATOR_VERSION = "1.0.0"
HASHING = "SHA-256 over UTF-8 bytes with CRLF normalised to LF"


def _reject_duplicates(pairs):
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON key")
    return dict(pairs)


def _reject_constant(value):
    raise ValueError(f"non-JSON numeric constant {value}")


def parse_json_text(text):
    """Strict JSON parse: no duplicate keys and no NaN/Infinity."""
    return json.loads(text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)


def serialize_json(value):
    """Canonical file text: 2-space indent, insertion order kept, UTF-8, LF, final newline."""
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def text_hash(text):
    return g1.sha256(text.encode("utf-8"))


def build_record(dataset_id, hashes, spec_texts, specs):
    """Provenance record: G1 input hashes and one hash per call spec (no timestamps)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "dataset_id": dataset_id,
        "generator": {"package": "fyp_evaluation.call_specs", "version": GENERATOR_VERSION},
        "hashing": HASHING,
        "g1_inputs": dict(hashes),
        "call_specs": {spec["call_id"]: text_hash(spec_texts[spec["call_id"]]) for spec in specs},
        "summary": {
            "call_specs": len(specs),
            "gold_entries_per_call": len(specs[0]["gold"]) if specs else 0,
            "edge_case_assignments": sum(len(spec["edge_cases"]) for spec in specs),
            "calls_with_edge_cases": sum(1 for spec in specs if spec["edge_cases"]),
            "calls_with_compliance_fails": sum(
                1 for spec in specs
                if any(e["expected"].get("expected_verdict") == "FAIL" for e in spec["gold"])),
        },
    }
