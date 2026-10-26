"""Python-owned compilation of the dialogue: turn IDs, event spans, length metrics and hashes are never model output.

A pure function of (version-matched inputs, accepted parsed response, recorded generation metadata). The length block
uses the length policy recorded in the generation params, so a historical dialogue rebuilds byte for byte after the
policy is recalibrated. TTS and transcript consumers read only `turns[].turn_id/speaker/text`.
"""

from fyp_evaluation import register_guide
from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json, text_hash
from fyp_evaluation.llm.canonical import canonical_response

from .config import GENERATOR_VERSION, HASHING, STAGE, schema_version, unit_of
from .render import length_count, render
from .validate_response import review_notes


def generation_block(generation, parsed, response_schema):
    return {
        "attempt_id": generation["attempt_id"], "model": generation["model"], "params": generation["params"],
        "prompt_template_sha256": generation["prompt_template_sha256"],
        "response_schema_sha256": generation["response_schema_sha256"],
        "accepted_response_sha256": text_hash(serialize_json(canonical_response(parsed, response_schema))),
    }


def compile_artifact(inputs, ctx, parsed, generation, response_schema):
    turns, alignment, errors = render(ctx, parsed)
    if errors:
        raise ValueError("cannot compile the dialogue: " + "; ".join(errors[:5]))
    policy = generation["params"]["length_policy"]
    low, high = policy["profiles"][ctx.profile]["accepted_words"]
    unit = unit_of(policy)
    words = length_count(turns, unit)
    call_id, metadata = ctx.call_id, ctx.metadata
    multilingual = ctx.multilingual
    meta_keys = ("language_profile", "mixing_style") if multilingual else ("language",)
    artifact = {
        "schema_version": schema_version(inputs.documents),
        "stage": STAGE,
        "dataset_id": inputs.dataset_id,
        "call_id": call_id,
        "call_spec_sha256": inputs.spec_hashes[call_id],
        "check_plan_skeleton_sha256": inputs.skeleton_hashes[call_id],
        "call_config_sha256": inputs.config_hashes[call_id],
        "check_plan_sha256": inputs.plan_hashes[call_id],
        "plan_verification_sha256": inputs.report_hashes[call_id],
        "source": {**inputs.source_hashes(), **(register_guide.recorded(generation) if multilingual else {})},
        "generation": generation_block(generation, parsed, response_schema),
        "metadata": {key: metadata[key] for key in (*meta_keys, "duration_profile", "target_duration_min_seconds",
                                                      "target_duration_max_seconds")},
        "length": {"policy_version": policy["version"], "assumed_wpm": policy["assumed_wpm"], "accepted_word_range": [low, high],
                   "words": words, "turns": len(turns), "estimated_seconds": round(words * 60 / policy["assumed_wpm"], 1)},
        "turns": turns,
        "event_alignment": sorted(alignment, key=lambda span: (span["turn_id"], span["start"])),
        "review_notes": review_notes(ctx, turns, alignment),
    }
    if multilingual:
        artifact["length"]["unit"] = unit
    return artifact


def build_record(inputs, artifact_texts, provenance_by_call):
    """Deterministic record over the accepted dialogues (no timestamps)."""
    calls, words, units, multilingual = {}, {}, {}, schema_version(inputs.documents) != "1.0"
    for call_id in inputs.call_ids:
        if call_id not in artifact_texts:
            continue
        artifact = parse_json_text(artifact_texts[call_id])
        length = artifact["length"]
        group = (f"{artifact['metadata']['language_profile']}/{artifact['metadata']['duration_profile']}" if multilingual
                 else artifact["metadata"]["duration_profile"])
        words.setdefault(group, []).append(length["words"])
        units[group] = length.get("unit")
        calls[call_id] = {
            "artifact_sha256": text_hash(artifact_texts[call_id]),
            "accepted_attempt_id": artifact["generation"]["attempt_id"],
            "turns": length["turns"], "words": length["words"], "estimated_seconds": length["estimated_seconds"],
            **({"unit": length["unit"]} if multilingual else {}),
            "review_notes": len(artifact["review_notes"]),
            "call_config_sha256": artifact["call_config_sha256"],
            "check_plan_sha256": artifact["check_plan_sha256"],
            "plan_verification_sha256": artifact["plan_verification_sha256"],
            "previous_artifact_sha256": [entry["artifact_sha256"] for entry in provenance_by_call[call_id]["accepted_history"]],
        }
    mean = {group: {"min": min(values), "max": max(values), "mean": round(sum(values) / len(values), 1)} for group, values in sorted(words.items())}
    if multilingual:  # counts are only comparable within one unit: state it per group
        mean = {group: {"unit": units[group], **row} for group, row in mean.items()}
    return {
        "schema_version": schema_version(inputs.documents),
        "stage": STAGE,
        "dataset_id": inputs.dataset_id,
        "generator": {"package": "fyp_evaluation.dialogues", "version": GENERATOR_VERSION},
        "hashing": HASHING,
        "upstream": {"plan_verification_record_sha256": inputs.report_record_sha256, **inputs.source_hashes()},
        "calls": calls,
        "summary": {
            "accepted_calls": len(calls), "total_calls": len(inputs.call_ids),
            "reopened_calls": sum(1 for entry in calls.values() if entry["previous_artifact_sha256"]),
            ("length_by_profile" if multilingual else "words_by_profile"): mean,
            "review_notes": sum(entry["review_notes"] for entry in calls.values()),
        },
    }
