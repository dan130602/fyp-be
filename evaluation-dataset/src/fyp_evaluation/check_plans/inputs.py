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

    def __init__(self, upstream, configs, config_texts, config_record_sha256, ready_call_ids=None):
        self.upstream = upstream  # call_configs.inputs.ConfigInputs (G0-G2B validated)
        self.root, self.dataset_id, self.call_ids = upstream.root, upstream.dataset_id, upstream.call_ids
        self.specs, self.spec_hashes, self.documents = upstream.specs, upstream.spec_hashes, upstream.documents
        self.skeleton_texts, self.skeleton_hashes = upstream.skeleton_texts, upstream.skeleton_hashes
        self.skeletons = {call_id: parse_json_text(text) for call_id, text in upstream.skeleton_texts.items()}
        self.configs, self.config_texts = configs, config_texts
        # Calls whose accepted G3 config is loaded: every call normally, only the selected calls in selected-call mode.
        self.ready_call_ids = list(self.call_ids) if ready_call_ids is None else list(ready_call_ids)
        self.config_hashes = {call_id: text_hash(text) for call_id, text in config_texts.items()}
        self.config_record_sha256 = config_record_sha256
        self.skeleton_record_sha256 = upstream.skeleton_record_sha256
        self.product_category, self.price_ranges = upstream.product_category, upstream.price_ranges
        self.currency_profile = upstream.currency_profile

    def source_hashes(self):
        """The G0 hashes G4 records. A multilingual release also records `language_profiles_sha256`: the fresh upstream hashes
        include it, so leaving it out would make every multilingual plan look stale."""
        hashes = self.upstream.source_hashes()
        language_profiles = self.upstream.upstream.g0_hashes.get("language_profiles_sha256")
        return {**hashes, "language_profiles_sha256": language_profiles} if language_profiles else hashes


def load_inputs(root, schema_dir=DEFAULT_SCHEMA_DIR, call_ids=None):
    """Refuse unless G0-G2B validate and the needed G3 configs are valid.

    Without `call_ids` G3 must be complete (`--validate --require-complete`). With `call_ids` (selected-call mode) G3 must
    still be intact release-wide (every existing artifact, provenance file and the record valid and current), but only the
    selected calls need an accepted config; unrelated calls may be missing. Every call that has an accepted config is loaded
    (`ready_call_ids`), so earlier plans stay validatable; calls without one are simply not eligible.
    """
    upstream = load_g3_inputs(root, schema_dir)  # raises for an invalid G0/G1/G2A/G2B
    selected = list(dict.fromkeys(call_ids)) if call_ids else None
    if selected and (unknown := [c for c in selected if c not in upstream.call_ids]):
        raise G3StateError(f"unknown call id(s): {', '.join(unknown)}")
    errors = ConfigRunner(root, schema_dir=schema_dir, inputs=upstream).validate(require_complete=selected is None)
    if errors:
        raise G3StateError("G3 call configs are not complete and valid; fix G3 first:\n" + "\n".join(errors[:20]))
    root = Path(root)
    texts, configs = {}, {}
    for call_id in upstream.call_ids:
        path = root / "authoring/call_configs" / f"{call_id}.json"
        if selected is not None and not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
        texts[call_id], configs[call_id] = text, parse_json_text(text)
    if selected and (missing := [c for c in selected if c not in configs]):
        raise G3StateError(f"no accepted G3 call config for the selected call(s): {', '.join(missing)}")
    record = root / G3_RECORD
    if not record.is_file():
        raise G3StateError("G3 has no call_config_record.json; fix G3 first")
    return PlanInputs(upstream, configs, texts, g1.sha256(record.read_bytes()), ready_call_ids=list(configs))


def inputs_with_config(inputs, call_id, text):
    """A copy of `inputs` in which one call's G3 config is replaced by `text` (a reconstructed historical version)."""
    clone = copy(inputs)
    clone.configs = {**inputs.configs, call_id: parse_json_text(text)}
    clone.config_texts = {**inputs.config_texts, call_id: text}
    clone.config_hashes = {**inputs.config_hashes, call_id: text_hash(text)}
    return clone
