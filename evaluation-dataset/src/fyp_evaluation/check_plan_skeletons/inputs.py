"""Gate on a valid G2A state, then load the immutable inputs of G2B.

G2B reads business truth only from the call specs and the frozen G0 documents. The G1
CSVs are read solely by the upstream validators that the gate runs; they are never a
source of values here.
"""

from pathlib import Path

from fyp_evaluation.call_specs import compile as g2a
from fyp_evaluation.call_specs.outputs import read_disk as read_g2a_disk
from fyp_evaluation.call_specs.validate import validate_spec_set as validate_g2a
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

G0_SOURCES = {
    "checklist_sha256": "checklist.json",
    "check_semantics_sha256": "check_semantics.json",
    "edge_case_catalog_sha256": "edge_case_catalog.json",
    "normalization_sha256": "normalization.json",
}


class G2AStateError(ValueError):
    """The G2A state on disk is not valid, so no skeleton may be built from it."""


class SkeletonInputs:
    """Frozen G0 documents plus the validated call specs, with every hash G2B records."""

    def __init__(self, root, documents, dataset_id, specs, spec_texts, g0_hashes, record_sha256):
        self.root, self.documents, self.dataset_id = Path(root), documents, dataset_id
        self.call_ids = list(specs)
        self.specs, self.spec_texts, self.g0_hashes = specs, spec_texts, g0_hashes
        self.spec_hashes = {call_id: g2a.text_hash(text) for call_id, text in spec_texts.items()}
        self.record_sha256 = record_sha256


def file_hash(path):
    return g1.sha256(Path(path).read_bytes())


def g0_hashes(root):
    return {key: file_hash(Path(root) / relative) for key, relative in G0_SOURCES.items()}


def load_inputs(root, schema_dir=DEFAULT_SCHEMA_DIR):
    """Refuse unless G0, G1 and G2A all validate on disk; then parse the call specs."""
    root = Path(root)
    src = g2a.load_g1_sources(root, schema_dir)  # raises G1StateError (a ValueError) when G0/G1 are invalid
    disk = read_g2a_disk(root)
    if errors := validate_g2a(src, disk, schema_dir):
        raise G2AStateError("G2A outputs are not valid; regenerate or fix G2A first:\n" + "\n".join(errors))
    spec_texts = {call_id: disk[g2a.spec_path(call_id)] for call_id in src.call_ids}
    specs = {call_id: g2a.parse_json_text(text) for call_id, text in spec_texts.items()}
    hashes = g0_hashes(root)
    for call_id, spec in specs.items():  # explicit chain: the checklist hash frozen through G2A is the one in use
        if spec["source"]["checklist_sha256"] != hashes["checklist_sha256"]:
            raise G2AStateError(f"{call_id}: checklist hash frozen in the call spec differs from checklist.json")
    return SkeletonInputs(root, src.documents, src.dataset_id, specs, spec_texts, hashes,
                          file_hash(root / g2a.RECORD))
