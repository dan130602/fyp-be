"""Offline fixtures for the schema-2.0 (multilingual) G4 tests. No network, no credentials, no paid calls.

G3 deliberately refuses the multilingual release until Phase 3, and that refusal stays in place. These helpers therefore
never go through G3: they build the validated G0-G2B upstream state directly, derive one *artificial* G3 call config per
call from G3's own deterministic synthetic response (never written to disk, never accepted by G3), and wrap it in
`PlanInputs`, which is everything the G4 context, analyser, request builder and compiler read.
"""

from pathlib import Path

from fyp_evaluation.call_configs.compile import compile_artifact as compile_config
from fyp_evaluation.call_configs.feasibility import synthesize_response as synthesize_config
from fyp_evaluation.call_configs.inputs import ConfigInputs
from fyp_evaluation.call_configs.request import response_schema_path
from fyp_evaluation.call_configs.slots import CallContext as ConfigContext
from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.check_plan_skeletons import inputs as g2b_inputs
from fyp_evaluation.check_plan_skeletons.compile import RECORD as SKELETON_RECORD, skeleton_path
from fyp_evaluation.check_plan_skeletons.outputs import read_disk
from fyp_evaluation.check_plans.context import PlanContext
from fyp_evaluation.check_plans.inputs import PlanInputs
from fyp_evaluation.planning import outputs as g1

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
RELEASE = ROOT / "datasets" / "advisory_execution_multilingual_v1"
PROFILES = ["ENGLISH", "MANDARIN", "EN_ZH_MIXED", "SINGLISH"]
_GENERATION = {"attempt_id": "A001", "model": "fixture", "prompt_template_sha256": "0" * 64, "response_schema_sha256": "0" * 64}

_CACHE = {}


def upstream_inputs(root=RELEASE):
    """G3-shaped upstream state of the release (G0-G2B validated on disk), bypassing the production G3 gate."""
    key = ("upstream", str(root))
    if key not in _CACHE:
        skeletons = g2b_inputs.load_inputs(root, SCHEMAS)
        disk = read_disk(root)
        texts = {call_id: disk[skeleton_path(call_id)] for call_id in skeletons.call_ids}
        _CACHE[key] = ConfigInputs(skeletons, texts, g1.sha256((Path(root) / SKELETON_RECORD).read_bytes()))
    return _CACHE[key]


def artificial_config(upstream, call_id, *, fee_basis="RATE"):
    """The text of an artificial G3 call config for one call (deterministic; never accepted by G3 or written to disk)."""
    schema = parse_json_text(response_schema_path(SCHEMAS, upstream.documents).read_text(encoding="utf-8"))
    ctx = ConfigContext(upstream, call_id)
    return serialize_json(compile_config(upstream, ctx, synthesize_config(ctx, fee_basis=fee_basis), _GENERATION, schema))


def plan_inputs(root=RELEASE, call_ids=None, *, fee_basis="RATE"):
    """`PlanInputs` for the multilingual release with artificial configs for `call_ids` (default: all calls)."""
    key = ("plan", str(root), tuple(call_ids) if call_ids else None, fee_basis)
    if key not in _CACHE:
        upstream = upstream_inputs(root)
        chosen = list(call_ids) if call_ids else list(upstream.call_ids)
        texts = {call_id: artificial_config(upstream, call_id, fee_basis=fee_basis) for call_id in chosen}
        configs = {call_id: parse_json_text(text) for call_id, text in texts.items()}
        _CACHE[key] = PlanInputs(upstream, configs, texts, "0" * 64, ready_call_ids=chosen)
    return _CACHE[key]


def context(call_id, *, fee_basis="RATE"):
    return PlanContext(plan_inputs(fee_basis=fee_basis), call_id)


def calls_of(profile):
    """Call IDs of one language profile, in release order."""
    import csv

    with (RELEASE / "authoring/call_manifest.csv").open(encoding="utf-8", newline="") as file:
        return [row["call_id"] for row in csv.DictReader(file) if row["language_profile"] == profile]
