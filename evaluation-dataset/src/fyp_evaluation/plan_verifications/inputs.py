"""Gate on a valid upstream state (G0-G3 complete and valid, G4 structurally valid), then load everything G5 reads.

G5 never re-implements G4's structural checks: it runs G4's own `--validate`. The only G4 findings it tolerates are the
ones explained by a per-call dependency change (a G3 config that changed after its plan was accepted); those calls are
merely ineligible for verification until G4 is regenerated. Every other G4 finding refuses the whole run.
"""

from copy import copy
from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text, text_hash
from fyp_evaluation.check_plans.inputs import load_inputs as load_g4_inputs
from fyp_evaluation.check_plans.run import CheckPlanRunner
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

G4_ARTIFACT_DIR = "authoring/check_plans"
G4_RECORD = "authoring/check_plan_record.json"


class G4StateError(ValueError):
    """G4 outputs are not valid on disk, so no G5 work may start."""


class VerificationInputs:
    """Validated upstream state (G0-G4) plus every hash G5 records."""

    def __init__(self, upstream, plan_texts, g4_stale, plan_record_sha256):
        self.upstream = upstream  # check_plans.inputs.PlanInputs (G0-G3 validated)
        self.root, self.dataset_id, self.call_ids = upstream.root, upstream.dataset_id, upstream.call_ids
        self.specs, self.spec_hashes, self.documents = upstream.specs, upstream.spec_hashes, upstream.documents
        self.skeleton_hashes = upstream.skeleton_hashes
        self.configs, self.config_texts, self.config_hashes = upstream.configs, upstream.config_texts, upstream.config_hashes
        self.plan_texts = plan_texts  # only calls with an accepted G4 plan
        self.plans = {call_id: parse_json_text(text) for call_id, text in plan_texts.items()}
        self.plan_hashes = {call_id: text_hash(text) for call_id, text in plan_texts.items()}
        self.g4_stale = g4_stale  # {call_id: [changed keys]}: accepted plans built on a G3 config that has since changed
        self.plan_record_sha256 = plan_record_sha256

    def source_hashes(self):
        return self.upstream.source_hashes()


def _blocking_g4_errors(g4, errors):
    """G4 --validate findings that are NOT explained by a per-call G3 dependency change."""
    stale = {call_id: info["changed"] for call_id, info in g4.dependency_status().items() if info["status"] == "stale"}
    tolerated_prefixes = tuple(f"{call_id}: stale " for call_id in stale)
    record_note = f"differs from the rebuild: {G4_RECORD}"
    blocking = [e for e in errors if not (e.startswith(tolerated_prefixes) or (stale and e == record_note))]
    return blocking, stale


def load_inputs(root, schema_dir=DEFAULT_SCHEMA_DIR):
    """Refuse unless G0-G3 are complete and valid and every G4 artifact is structurally valid and reproducible."""
    upstream = load_g4_inputs(root, schema_dir)  # raises for an invalid G0-G2B or an incomplete/invalid G3
    g4 = CheckPlanRunner(root, schema_dir=schema_dir, inputs=upstream)
    blocking, stale = _blocking_g4_errors(g4, g4.validate())
    if blocking:
        raise G4StateError("G4 check plans are not valid; fix G4 first:\n" + "\n".join(blocking[:20]))
    root = Path(root)
    texts = {}
    for call_id in upstream.call_ids:
        path = root / G4_ARTIFACT_DIR / f"{call_id}.json"
        if path.is_file():
            texts[call_id] = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    record = root / G4_RECORD
    return VerificationInputs(upstream, texts, stale, g1.sha256(record.read_bytes()) if record.is_file() else None)


def fresh_upstream_hashes(root, call_id):
    """The hashes G5 records for one call, re-read from the files on disk right now (raises OSError if one is missing)."""
    from fyp_evaluation.call_configs.inputs import fresh_upstream_hashes as g3_fresh

    root = Path(root)
    hashes = g3_fresh(root, call_id)
    hashes["call_config"] = g1.sha256((root / "authoring/call_configs" / f"{call_id}.json").read_bytes())
    hashes["check_plan"] = g1.sha256((root / G4_ARTIFACT_DIR / f"{call_id}.json").read_bytes())
    return hashes


def inputs_with_versions(inputs, call_id, config_text, plan_text):
    """A copy of `inputs` in which one call's G3 config and G4 plan are replaced by reconstructed historical versions."""
    clone = copy(inputs)
    clone.configs = {**inputs.configs, call_id: parse_json_text(config_text)}
    clone.config_texts = {**inputs.config_texts, call_id: config_text}
    clone.config_hashes = {**inputs.config_hashes, call_id: text_hash(config_text)}
    clone.plan_texts = {**inputs.plan_texts, call_id: plan_text}
    clone.plans = {**inputs.plans, call_id: parse_json_text(plan_text)}
    clone.plan_hashes = {**inputs.plan_hashes, call_id: text_hash(plan_text)}
    return clone
