"""Compile the accepted call_config artifact and its record from a validated parsed response.

Pure functions of (frozen inputs, accepted parsed response, recorded generation metadata): `--check` can rebuild
every artifact byte for byte without any provider envelope.
"""

from fyp_evaluation.call_specs.compile import serialize_json, text_hash
from fyp_evaluation.llm.canonical import canonical_response, ordered, response_hash  # noqa: F401  (re-exported)

from .config import GENERATOR_VERSION, HASHING, SCHEMA_VERSION, STAGE


def compile_artifact(inputs, ctx, parsed, generation, response_schema):
    """The Python-owned envelope around a validated scenario. `generation` = attempt/model/template/schema hashes."""
    scenario = canonical_response(parsed, response_schema)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "dataset_id": inputs.dataset_id,
        "call_id": ctx.call_id,
        "call_spec_sha256": inputs.spec_hashes[ctx.call_id],
        "check_plan_skeleton_sha256": inputs.skeleton_hashes[ctx.call_id],
        "source": inputs.source_hashes(),
        "generation": {
            "attempt_id": generation["attempt_id"], "model": generation["model"],
            "prompt_template_sha256": generation["prompt_template_sha256"],
            "response_schema_sha256": generation["response_schema_sha256"],
            "accepted_response_sha256": text_hash(serialize_json(scenario)),
        },
        "products": [dict(product) for product in ctx.products],
        "scenario": scenario,
    }


def build_record(inputs, artifact_texts, provenance_by_call):
    """Deterministic record over the accepted artifacts (no timestamps)."""
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
            "previous_artifact_sha256": [entry["artifact_sha256"] for entry in history],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "dataset_id": inputs.dataset_id,
        "generator": {"package": "fyp_evaluation.call_configs", "version": GENERATOR_VERSION},
        "hashing": HASHING,
        "upstream": {
            "call_spec_record_sha256": inputs.upstream.record_sha256,
            "check_plan_skeleton_record_sha256": inputs.skeleton_record_sha256,
            **inputs.source_hashes(),
        },
        "calls": calls,
        "summary": {
            "accepted_calls": len(calls),
            "total_calls": len(inputs.call_ids),
            "reopened_calls": sum(1 for entry in calls.values() if entry["previous_artifact_sha256"]),
        },
    }
