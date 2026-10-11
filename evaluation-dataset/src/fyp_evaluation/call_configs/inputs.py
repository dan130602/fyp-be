"""Gate on a valid G2B state, then load everything G3 reads.

G3 sends the model a reduced brief built from the call spec and G0 only. It never reads the skeleton body for
content; the skeleton is validated (so G3 cannot be built on stale structure) and its hash is recorded.
"""

from pathlib import Path

from fyp_evaluation.call_specs import compile as g2a
from fyp_evaluation.call_specs.compile import parse_json_text, text_hash
from fyp_evaluation.check_plan_skeletons import inputs as g2b_inputs
from fyp_evaluation.check_plan_skeletons.compile import RECORD as SKELETON_RECORD, skeleton_path
from fyp_evaluation.check_plan_skeletons.outputs import read_disk as read_g2b_disk
from fyp_evaluation.check_plan_skeletons.validate import validate_skeleton_set as validate_g2b
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR


class G2BStateError(ValueError):
    """The G2B state on disk is not valid, so no G3 work may start."""


class ConfigInputs:
    """Validated upstream state plus the hashes G3 records."""

    def __init__(self, upstream, skeleton_texts, skeleton_record_sha256):
        self.upstream = upstream  # check_plan_skeletons.inputs.SkeletonInputs (G0 + G1 + G2A validated)
        self.root, self.dataset_id, self.call_ids = upstream.root, upstream.dataset_id, upstream.call_ids
        self.specs, self.spec_hashes, self.documents = upstream.specs, upstream.spec_hashes, upstream.documents
        self.skeleton_texts = skeleton_texts
        self.skeleton_hashes = {call_id: text_hash(text) for call_id, text in skeleton_texts.items()}
        self.skeleton_record_sha256 = skeleton_record_sha256
        plan = self.documents["generation_plan"]
        self.generation_plan_sha256 = g1.sha256((Path(self.root) / "authoring/generation_plan.json").read_bytes())
        extraction = plan["extraction_distribution"]
        self.product_category = {item["name"]: item["category"] for item in extraction["product_pool"]}
        self.price_ranges = extraction["amount_quantity_coherence"]["ranges"]
        self.currency_profile = self.documents["normalization"]["profiles"]["currency_iso_4217_alpha3"]

    def source_hashes(self):
        """The five hashes of frozen inputs that G3 reads (all verified by the upstream gate)."""
        hashes = self.upstream.g0_hashes
        return {
            "checklist_sha256": hashes["checklist_sha256"],
            "check_semantics_sha256": hashes["check_semantics_sha256"],
            "edge_case_catalog_sha256": hashes["edge_case_catalog_sha256"],
            "normalization_sha256": hashes["normalization_sha256"],
            "generation_plan_sha256": self.generation_plan_sha256,
        }


def load_inputs(root, schema_dir=DEFAULT_SCHEMA_DIR):
    """Refuse unless G0, G1, G2A and G2B all validate on disk."""
    upstream = g2b_inputs.load_inputs(root, schema_dir)  # raises for an invalid G0/G1/G2A
    disk = read_g2b_disk(root)
    if errors := validate_g2b(upstream, disk, schema_dir):
        raise G2BStateError("G2B outputs are not valid; regenerate or fix G2B first:\n" + "\n".join(errors))
    texts = {call_id: disk[skeleton_path(call_id)] for call_id in upstream.call_ids}
    for call_id, text in texts.items():  # explicit chain: the skeleton was compiled from this exact call spec
        if parse_json_text(text)["call_spec_sha256"] != upstream.spec_hashes[call_id]:
            raise G2BStateError(f"{call_id}: skeleton was built from a different call spec")
    return ConfigInputs(upstream, texts, g1.sha256((Path(root) / SKELETON_RECORD).read_bytes()))


def fresh_upstream_hashes(root, call_id):
    """The hashes G3 records for one call (spec, skeleton, five G0 files), re-read from the files on disk right now.

    Same keys and same hashing as `ConfigRunner._upstream_hashes`, but taken from disk instead of cached inputs.
    Raises OSError if an upstream file is missing or unreadable.
    """
    root = Path(root)
    hashes = {
        "call_spec": g1.sha256((root / g2a.spec_path(call_id)).read_bytes()),
        "check_plan_skeleton": g1.sha256((root / skeleton_path(call_id)).read_bytes()),
    }
    hashes.update(g2b_inputs.g0_hashes(root))
    hashes["generation_plan_sha256"] = g1.sha256((root / "authoring/generation_plan.json").read_bytes())
    return hashes
