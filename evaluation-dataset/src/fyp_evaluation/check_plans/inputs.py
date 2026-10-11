"""Gate on valid upstream state (G0-G2B and a complete, valid G3), then load everything G4 reads.

G4 reads business truth only from the frozen G2B skeletons, the accepted G3 call configs and G0 semantics. The skeleton
is never modified; the G3 hashes are recorded so a later G3 reopen marks the dependent plan stale.
"""

from copy import copy
from pathlib import Path

from fyp_evaluation.call_configs.inputs import load_inputs as load_g3_inputs
from fyp_evaluation.call_configs.run import ConfigRunner
from fyp_evaluation.call_specs.compile import parse_json_text, text_hash
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

G3_RECORD = "authoring/call_config_record.json"


class HistoricalContextError(ValueError):
    """The upstream version a G4 plan was built against cannot be reconstructed; never replaced by the current inputs."""


class G3StateError(ValueError):
    """G3 is not complete and valid on disk, so no G4 work may start."""


class PlanInputs:
    """Validated upstream state plus every hash G4 records."""

    def __init__(self, upstream, configs, config_texts, config_record_sha256):
        self.upstream = upstream  # call_configs.inputs.ConfigInputs (G0-G2B validated)
        self.root, self.dataset_id, self.call_ids = upstream.root, upstream.dataset_id, upstream.call_ids
        self.specs, self.spec_hashes, self.documents = upstream.specs, upstream.spec_hashes, upstream.documents
        self.skeleton_texts, self.skeleton_hashes = upstream.skeleton_texts, upstream.skeleton_hashes
        self.skeletons = {call_id: parse_json_text(text) for call_id, text in upstream.skeleton_texts.items()}
        self.configs, self.config_texts = configs, config_texts
        self.config_hashes = {call_id: text_hash(text) for call_id, text in config_texts.items()}
        self.config_record_sha256 = config_record_sha256
        self.skeleton_record_sha256 = upstream.skeleton_record_sha256
        self.product_category, self.price_ranges = upstream.product_category, upstream.price_ranges
        self.currency_profile = upstream.currency_profile

    def source_hashes(self):
        return self.upstream.source_hashes()


def load_inputs(root, schema_dir=DEFAULT_SCHEMA_DIR):
    """Refuse unless G0-G2B validate and G3 is complete and valid (`--validate --require-complete`)."""
    upstream = load_g3_inputs(root, schema_dir)  # raises for an invalid G0/G1/G2A/G2B
    errors = ConfigRunner(root, schema_dir=schema_dir, inputs=upstream).validate(require_complete=True)
    if errors:
        raise G3StateError("G3 call configs are not complete and valid; fix G3 first:\n" + "\n".join(errors[:20]))
    root = Path(root)
    texts, configs = {}, {}
    for call_id in upstream.call_ids:
        text = (root / "authoring/call_configs" / f"{call_id}.json").read_bytes().decode("utf-8").replace("\r\n", "\n")
        texts[call_id], configs[call_id] = text, parse_json_text(text)
    return PlanInputs(upstream, configs, texts, g1.sha256((root / G3_RECORD).read_bytes()))


def inputs_with_config(inputs, call_id, text):
    """A copy of `inputs` in which one call's G3 config is replaced by `text` (a reconstructed historical version)."""
    clone = copy(inputs)
    clone.configs = {**inputs.configs, call_id: parse_json_text(text)}
    clone.config_texts = {**inputs.config_texts, call_id: text}
    clone.config_hashes = {**inputs.config_hashes, call_id: text_hash(text)}
    return clone
