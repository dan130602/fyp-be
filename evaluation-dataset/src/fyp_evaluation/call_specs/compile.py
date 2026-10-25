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
from fyp_evaluation.validation.stage0 import (
    DEFAULT_SCHEMA_DIR, LANGUAGE_PROFILES_FILE, is_multilingual, load_stage0, validate_stage0_release,
)

SCHEMA_VERSION = "1.0"
# Multilingual releases freeze the language profile, its resolved requirements and the edge-case language variants.
MULTILINGUAL_SCHEMA_VERSION = "2.0"
MANIFEST_META_V1 = ("call_intents", "language", "duration_profile", "target_duration_min_seconds", "target_duration_max_seconds")
MANIFEST_META_V2 = ("call_intents", "language_profile", "mixing_style", "duration_profile",
                    "target_duration_min_seconds", "target_duration_max_seconds")
EDGE_KEYS_V1 = ("assignment_id", "edge_case_id", "family", "target_question_ids", "parameters")
EDGE_KEYS_V2 = ("assignment_id", "edge_case_id", "family", "target_question_ids", "language_variant", "parameters")
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
        self.multilingual = is_multilingual(documents)
        self.schema_version = MULTILINGUAL_SCHEMA_VERSION if self.multilingual else SCHEMA_VERSION
        self.meta_keys = MANIFEST_META_V2 if self.multilingual else MANIFEST_META_V1
        self.edge_keys = EDGE_KEYS_V2 if self.multilingual else EDGE_KEYS_V1


def read_sources(root):
    """Parse the G1 files and hash them, with no validation (callers gate first)."""
    root = Path(root)
    documents = load_stage0(root)
    texts = {rel: (root / rel).read_text(encoding="utf-8") for rel in (g1.MANIFEST, g1.MATRIX, g1.ASSIGNMENTS)}
    hashes = {key: g1.sha256((root / rel).read_bytes()) for key, rel in G1_SOURCES.items()}
    hashes["checklist_sha256"] = g1.sha256((root / CHECKLIST).read_bytes())
    if "language_profiles" in documents:
        hashes["language_profiles_sha256"] = g1.sha256((root / LANGUAGE_PROFILES_FILE).read_bytes())
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


def language_requirements(src, profile, mixing_style):
    """The frozen snapshot of one language profile's requirements (resolved from language_profiles.json)."""
    definition = src.documents["language_profiles"]["profiles"][profile]
    requirements = {key: definition[key] for key in ("value_surface_languages", "product_name_languages", "script_rules", "speakers")}
    if "mixing_styles" in definition:
        requirements["mixing_style_description"] = definition["mixing_styles"][mixing_style]
    return requirements


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
        spec = {
            "schema_version": src.schema_version,
            "stage": STAGE,
            "dataset_id": src.dataset_id,
            "call_id": call_id,
            "metadata": {key: deepcopy(row[key]) for key in src.meta_keys},
        }
        if src.multilingual:
            spec["language_requirements"] = language_requirements(src, row["language_profile"], row["mixing_style"])
        spec["gold"] = gold_entries(src, call_id)
        spec["edge_cases"] = [{key: deepcopy(a[key]) for key in src.edge_keys} for a in by_call.get(call_id, [])]
        spec["source"] = dict(src.hashes)
        specs.append(spec)
    return specs


def project_to_g1_texts(specs, documents):
    """Rebuild the three G1 CSV texts from specs using G1's own canonical serializer."""
    calls, gold, assignments = [], {}, []
    for spec in specs:
        meta, call_id = spec["metadata"], spec["call_id"]
        calls.append({"call_id": call_id, **meta})
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
    multilingual = "language_profiles_sha256" in hashes
    record = {
        "schema_version": MULTILINGUAL_SCHEMA_VERSION if multilingual else SCHEMA_VERSION,
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
    if multilingual:
        profiles = {}
        for spec in specs:
            profile = spec["metadata"]["language_profile"]
            profiles[profile] = profiles.get(profile, 0) + 1
        record["summary"]["calls_by_language_profile"] = profiles
        record["summary"]["edge_case_language_variants"] = sum(
            1 for spec in specs for e in spec["edge_cases"] if e["language_variant"] != "NONE")
    return record
