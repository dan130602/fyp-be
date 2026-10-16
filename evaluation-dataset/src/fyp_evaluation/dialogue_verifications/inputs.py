"""Everything G7 reads: G0-G5 as G6 gates them, plus the accepted G6 dialogue of every call.

G7 never re-validates G0-G5 itself: it reuses G6's input gate unchanged. A dialogue is read one call at a time through
`read_dialogue` (the read is the only classifier): a missing or unreadable dialogue only makes that call ineligible (the
per-call integrity gate lives in `DialogueRunner.g7_eligibility`), never the whole stage.
"""

from copy import copy
from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text
from fyp_evaluation.dialogues.inputs import fresh_upstream_hashes as g6_fresh
from fyp_evaluation.dialogues.inputs import load_inputs as load_g6_inputs
from fyp_evaluation.dialogues.inputs import read_dialogue
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

G6_RECORD = "authoring/dialogue_record.json"


class DialogueVerificationInputs:
    """Validated upstream state (G0-G5, via G6) plus the dialogue fingerprint and parsed dialogue of every readable call."""

    def __init__(self, upstream, dialogue_hashes, dialogue_texts, g6_record_sha256):
        self.upstream = upstream  # dialogues.inputs.DialogueInputs
        self.root, self.dataset_id, self.call_ids = upstream.root, upstream.dataset_id, upstream.call_ids
        self.specs, self.spec_hashes, self.documents = upstream.specs, upstream.spec_hashes, upstream.documents
        self.skeleton_hashes = upstream.skeleton_hashes
        self.configs, self.config_hashes = upstream.configs, upstream.config_hashes
        self.plans, self.plan_hashes = upstream.plans, upstream.plan_hashes
        self.report_hashes = upstream.report_hashes
        self.dialogue_hashes = dialogue_hashes  # {call_id: sha256 | None (absent) | DIALOGUE_UNREADABLE}
        self.dialogue_texts = dialogue_texts  # {call_id: text} for every readable dialogue
        self.dialogues = {}
        for call_id, text in dialogue_texts.items():
            try:
                self.dialogues[call_id] = parse_json_text(text)
            except ValueError:
                pass  # not JSON: the per-call integrity gate reports it, and no context is built for the call
        self.g6_record_sha256 = g6_record_sha256

    def source_hashes(self):
        return self.upstream.source_hashes()


def load_inputs(root, schema_dir=DEFAULT_SCHEMA_DIR):
    """Refuse unless G0-G4 pass G5's gate (as G6 applies it); dialogues are read per call and never refuse the stage."""
    upstream = load_g6_inputs(root, schema_dir)
    hashes, texts = {}, {}
    for call_id in upstream.call_ids:
        hashes[call_id], text = read_dialogue(root, call_id)
        if text is not None:
            texts[call_id] = text
    try:
        record = g1.sha256((Path(root) / G6_RECORD).read_bytes())
    except OSError:
        record = None
    return DialogueVerificationInputs(upstream, hashes, texts, record)


def fresh_upstream_hashes(root, call_id):
    """The hashes G7 records for one call, re-read from disk now. G0-G4 read errors raise; the G5 report and the dialogue never do."""
    hashes = g6_fresh(root, call_id)
    hashes["dialogue"] = read_dialogue(root, call_id)[0]
    return hashes


def inputs_with_versions(inputs, call_id, upstream, dialogue_hash, dialogue_text):
    """A copy of `inputs` bound to a reconstructed G6 input state (`upstream`) and the historical dialogue a report verified."""
    clone = copy(inputs)
    clone.upstream = upstream
    clone.configs, clone.config_hashes = upstream.configs, upstream.config_hashes
    clone.plans, clone.plan_hashes = upstream.plans, upstream.plan_hashes
    clone.report_hashes = upstream.report_hashes
    clone.dialogue_hashes = {**inputs.dialogue_hashes, call_id: dialogue_hash}
    clone.dialogue_texts = {**inputs.dialogue_texts, call_id: dialogue_text}
    clone.dialogues = {**inputs.dialogues, call_id: parse_json_text(dialogue_text)}
    return clone
