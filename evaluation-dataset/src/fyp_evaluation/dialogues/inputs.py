"""Gate on a valid upstream state (G0-G4, exactly as G5 gates it), then load the G5 report fingerprint of every call.

G6 never re-validates G0-G4 itself: it reuses G5's input gate unchanged. G5 reports are read one call at a time: a missing
or unreadable report only makes that call ineligible (the G5 gate itself lives in `run.py`), never the whole stage.
"""

from copy import copy
from pathlib import Path

from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.plan_verifications.inputs import fresh_upstream_hashes as g5_fresh
from fyp_evaluation.plan_verifications.inputs import load_inputs as load_g5_inputs
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .config import ARTIFACT_DIR, DIALOGUE_UNREADABLE, G5_ARTIFACT_DIR, G5_RECORD, G5_UNREADABLE


def g5_report_fingerprint(root, call_id):
    """sha256 of the call's G5 report, None when it is confirmed absent, or G5_UNREADABLE. Never raises.

    The read itself is the only classifier: `exists()`/`is_file()` report False for any OSError, which would turn an
    access failure into a confirmed deletion.
    """
    try:
        data = (Path(root) / G5_ARTIFACT_DIR / f"{call_id}.json").read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        return G5_UNREADABLE
    return g1.sha256(data)


def read_dialogue(root, call_id):
    """(fingerprint, text): sha256 and CRLF-normalised text of the call's G6 dialogue, (None, None) when it is confirmed
    absent, or (DIALOGUE_UNREADABLE, None). Never raises; the read itself is the only classifier (see g5_report_fingerprint)."""
    try:
        data = (Path(root) / ARTIFACT_DIR / f"{call_id}.json").read_bytes()
    except FileNotFoundError:
        return None, None
    except OSError:
        return DIALOGUE_UNREADABLE, None
    try:
        return g1.sha256(data), data.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError:
        return g1.sha256(data), None  # readable bytes that are not text: the integrity gate reports it


def dialogue_fingerprint(root, call_id):
    """sha256 of the call's G6 dialogue, None when confirmed absent, or DIALOGUE_UNREADABLE. Never raises."""
    return read_dialogue(root, call_id)[0]


class DialogueInputs:
    """Validated upstream state (G0-G4) plus the G5 report fingerprint of every call."""

    def __init__(self, upstream, report_hashes, report_record_sha256):
        self.upstream = upstream  # plan_verifications.inputs.VerificationInputs
        self.root, self.dataset_id, self.call_ids = upstream.root, upstream.dataset_id, upstream.call_ids
        self.specs, self.spec_hashes, self.documents = upstream.specs, upstream.spec_hashes, upstream.documents
        self.skeleton_hashes = upstream.skeleton_hashes
        self.configs, self.config_hashes = upstream.configs, upstream.config_hashes
        self.plans, self.plan_hashes = upstream.plans, upstream.plan_hashes
        self.ready_call_ids = list(upstream.plans)  # calls whose G3 config and G4 plan are accepted (all of them unless selected)
        self.report_hashes = report_hashes  # {call_id: sha256 | None (absent) | G5_UNREADABLE}
        self.report_record_sha256 = report_record_sha256

    def source_hashes(self):
        return self.upstream.source_hashes()


def load_inputs(root, schema_dir=DEFAULT_SCHEMA_DIR, call_ids=None):
    """Refuse unless G0-G4 pass G5's own gate; G5 reports are fingerprinted per call and never refuse the stage.

    With `call_ids` (selected-call mode) G3 must still be intact release-wide, but only the selected calls need an accepted G3
    config and G4 plan; unrelated calls may be missing and are simply not eligible (G5's gate, unchanged)."""
    upstream = load_g5_inputs(root, schema_dir, call_ids) if call_ids else load_g5_inputs(root, schema_dir)
    reports = {call_id: g5_report_fingerprint(root, call_id) for call_id in upstream.call_ids}
    try:
        record = g1.sha256((Path(root) / G5_RECORD).read_bytes())
    except OSError:
        record = None
    return DialogueInputs(upstream, reports, record)


def fresh_upstream_hashes(root, call_id):
    """The hashes G6 records for one call, re-read from disk now. G0-G4 read errors raise; the G5 report never does."""
    hashes = g5_fresh(root, call_id)
    hashes["plan_verification"] = g5_report_fingerprint(root, call_id)
    return hashes


def inputs_with_versions(inputs, call_id, upstream, report_hash):
    """A copy of `inputs` bound to reconstructed G3/G4 versions (`upstream`) and the G5 report hash a dialogue recorded."""
    clone = copy(inputs)
    clone.upstream = upstream
    clone.configs, clone.config_hashes = upstream.configs, upstream.config_hashes
    clone.plans, clone.plan_hashes = upstream.plans, upstream.plan_hashes
    clone.report_hashes = {**inputs.report_hashes, call_id: report_hash}
    return clone
