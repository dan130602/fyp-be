"""Python-owned compilation of the check plan: frozen skeleton fields are copied, never accepted from the model.

`assemble` deep-copies the skeleton and attaches only the G4-added `authoring` blocks plus `derived_constraints`.
`project_check_plan_to_skeleton` removes exactly those additions; its output must serialise to the canonical frozen
skeleton bytes. Both are pure functions of (frozen inputs, accepted parsed response, recorded generation metadata), so
`--check` rebuilds every artifact without any provider envelope.
"""

from copy import deepcopy

from fyp_evaluation import register_guide
from fyp_evaluation.call_specs.compile import serialize_json, text_hash
from fyp_evaluation.llm.canonical import canonical_response

from .analysis import analyze
from .config import GENERATOR_VERSION, HASHING, STAGE, schema_version
from .template import Lexicon

FROZEN_BODY_KEYS = ("planned_events", "checks", "constraints")
G4_ONLY_TOP_LEVEL = ("check_plan_skeleton_sha256", "call_config_sha256", "generation", "derived_constraints")


def get_lexicon(inputs):
    if getattr(inputs, "_g4_lexicon", None) is None:
        inputs._g4_lexicon = Lexicon(inputs)
    return inputs._g4_lexicon


def assemble(ctx, authoring, derived, generation):
    """The complete check plan. Key order is fixed: identity, upstream hashes, generation, frozen copy, G4 additions."""
    inputs, skeleton = ctx.inputs, ctx.skeleton
    return {
        "schema_version": schema_version(inputs.documents),
        "stage": STAGE,
        "dataset_id": inputs.dataset_id,
        "call_id": ctx.call_id,
        "call_spec_sha256": inputs.spec_hashes[ctx.call_id],
        "check_plan_skeleton_sha256": inputs.skeleton_hashes[ctx.call_id],
        "call_config_sha256": inputs.config_hashes[ctx.call_id],
        "source": {**inputs.source_hashes(), **register_guide.extra_hashes(ctx.spec)},  # the guide only for SINGLISH calls
        "generation": dict(generation),
        "skeleton_header": {key: deepcopy(value) for key, value in skeleton.items() if key not in FROZEN_BODY_KEYS},
        "planned_events": [{**deepcopy(event), "authoring": authoring[event["event_id"]]} for event in skeleton["planned_events"]],
        "checks": deepcopy(skeleton["checks"]),
        "constraints": deepcopy(skeleton["constraints"]),
        "derived_constraints": derived,
    }


def project_check_plan_to_skeleton(plan):
    """Remove every G4-added field; what remains is the G2B skeleton, array order and key order included."""
    events = []
    for event in plan["planned_events"]:
        if "authoring" not in event:
            raise ValueError(f"{event.get('event_id')}: event has no authoring block")
        events.append({key: value for key, value in event.items() if key != "authoring"})
    return {**plan["skeleton_header"], "planned_events": events, "checks": plan["checks"], "constraints": plan["constraints"]}


def projection_errors(ctx, plan):
    try:
        projected = serialize_json(project_check_plan_to_skeleton(plan))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        return [f"projection failed: {error!r}"]
    if projected != ctx.inputs.skeleton_texts[ctx.call_id]:
        return ["the plan does not project back to the canonical frozen skeleton (a frozen field was altered)"]
    return []


def generation_block(generation, parsed, response_schema):
    return {
        "attempt_id": generation["attempt_id"], "model": generation["model"],
        "prompt_template_sha256": generation["prompt_template_sha256"],
        "response_schema_sha256": generation["response_schema_sha256"],
        "accepted_response_sha256": text_hash(serialize_json(canonical_response(parsed, response_schema))),
    }


def compile_artifact(inputs, ctx, parsed, generation, response_schema):
    authoring, derived, errors = analyze(ctx, get_lexicon(inputs), parsed)
    if errors:
        raise ValueError("cannot compile the check plan: " + "; ".join(errors[:5]))
    plan = assemble(ctx, authoring, derived, generation_block(generation, parsed, response_schema))
    if errors := projection_errors(ctx, plan):
        raise ValueError("; ".join(errors))
    return plan


def build_record(inputs, artifact_texts, provenance_by_call):
    """Deterministic record over the accepted plans (no timestamps)."""
    from fyp_evaluation.call_specs.compile import parse_json_text

    calls = {}
    for call_id in inputs.call_ids:
        if call_id not in artifact_texts:
            continue
        artifact = parse_json_text(artifact_texts[call_id])
        history = provenance_by_call[call_id]["accepted_history"]
        calls[call_id] = {
            "artifact_sha256": text_hash(artifact_texts[call_id]),
            "accepted_attempt_id": artifact["generation"]["attempt_id"],
            "call_spec_sha256": artifact["call_spec_sha256"],
            "check_plan_skeleton_sha256": artifact["check_plan_skeleton_sha256"],
            "call_config_sha256": artifact["call_config_sha256"],
            "previous_artifact_sha256": [entry["artifact_sha256"] for entry in history],
        }
    return {
        "schema_version": schema_version(inputs.documents),
        "stage": STAGE,
        "dataset_id": inputs.dataset_id,
        "generator": {"package": "fyp_evaluation.check_plans", "version": GENERATOR_VERSION},
        "hashing": HASHING,
        "upstream": {
            "call_spec_record_sha256": inputs.upstream.upstream.record_sha256,
            "check_plan_skeleton_record_sha256": inputs.skeleton_record_sha256,
            "call_config_record_sha256": inputs.config_record_sha256,
            **inputs.source_hashes(),
        },
        "calls": calls,
        "summary": {
            "accepted_calls": len(calls),
            "total_calls": len(inputs.call_ids),
            "reopened_calls": sum(1 for entry in calls.values() if entry["previous_artifact_sha256"]),
            "planned_events": sum(len(parse_json_text(artifact_texts[c])["planned_events"]) for c in calls),
        },
    }
