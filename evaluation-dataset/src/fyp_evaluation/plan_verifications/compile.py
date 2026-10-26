"""Python-owned compilation of a G5 report: identity, upstream hashes, expanded evidence and the derived verdict.

The model never states a verdict. Precedence: any MISMATCH (check or issue) -> INVALID; otherwise any UNCLEAR -> UNCLEAR;
otherwise VALID. The recommended repair is derived from the issues' advisory scopes and is never executed here.
Compilation is a pure function of (inputs bound to the verified G3/G4 versions, accepted parsed response, recorded
generation metadata), so `--check` rebuilds every report without any provider envelope.
"""

from fyp_evaluation import register_guide
from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json, text_hash
from fyp_evaluation.llm.canonical import canonical_response

from .config import GENERATOR_VERSION, HASHING, STAGE, VERDICTS, schema_version
from .validate_response import expand


def derive_verdict(checks, issues):
    assessments = [c["assessment"] for c in checks] + [i["assessment"] for i in issues]
    if "MISMATCH" in assessments:
        return "INVALID"
    if "UNCLEAR" in assessments:
        return "UNCLEAR"
    return "VALID"


def derive_repair(verdict, issues):
    if verdict == "VALID":
        return None
    if verdict == "UNCLEAR":
        return "REVIEW"
    scopes = {i["repair_scope"] for i in issues if i["assessment"] == "MISMATCH"}
    if "G3_PLUS_G4" in scopes:
        return "G3_PLUS_G4"
    if "REVIEW" in scopes:
        return "REVIEW"
    return "G4_ONLY"


def generation_block(generation, parsed, response_schema):
    return {
        "attempt_id": generation["attempt_id"], "model": generation["model"], "params": dict(generation["params"]),
        "prompt_template_sha256": generation["prompt_template_sha256"],
        "response_schema_sha256": generation["response_schema_sha256"],
        "accepted_response_sha256": text_hash(serialize_json(canonical_response(parsed, response_schema))),
    }


def compile_artifact(inputs, ctx, parsed, generation, response_schema):
    checks, issues, errors = expand(parsed, ctx)
    if errors:
        raise ValueError("cannot compile the plan verification: " + "; ".join(errors[:5]))
    verdict = derive_verdict(checks, issues)
    by_code = {}
    for issue in issues:
        by_code[issue["code"]] = by_code.get(issue["code"], 0) + 1
    call_id = ctx.call_id
    return {
        "schema_version": schema_version(inputs.documents),
        "stage": STAGE,
        "dataset_id": inputs.dataset_id,
        "call_id": call_id,
        "call_spec_sha256": inputs.spec_hashes[call_id],
        "check_plan_skeleton_sha256": inputs.skeleton_hashes[call_id],
        "call_config_sha256": inputs.config_hashes[call_id],
        "check_plan_sha256": inputs.plan_hashes[call_id],
        "source": {**inputs.source_hashes(), **register_guide.extra_hashes(ctx.spec)},  # the guide only for SINGLISH calls
        "generation": generation_block(generation, parsed, response_schema),
        "verdict": verdict,
        "recommended_repair": derive_repair(verdict, issues),
        "summary": {"match": sum(1 for c in checks if c["assessment"] == "MATCH"),
                    "mismatch": sum(1 for c in checks if c["assessment"] == "MISMATCH"),
                    "unclear": sum(1 for c in checks if c["assessment"] == "UNCLEAR"),
                    "issues_by_code": dict(sorted(by_code.items()))},
        "checks": checks,
        "issues": issues,
    }


def build_record(inputs, artifact_texts, provenance_by_call):
    """Deterministic record over the accepted reports (no timestamps)."""
    calls, verdicts, tier_not_echoed = {}, dict.fromkeys(VERDICTS, 0), 0
    for call_id in inputs.call_ids:
        if call_id not in artifact_texts:
            continue
        artifact = parse_json_text(artifact_texts[call_id])
        provenance = provenance_by_call[call_id]
        accepted_id = artifact["generation"]["attempt_id"]
        summary = next((a for a in provenance["attempts"] if a["attempt_id"] == accepted_id), {})
        observed = summary.get("service_tier")
        tier_not_echoed += observed is None
        verdicts[artifact["verdict"]] += 1
        calls[call_id] = {
            "artifact_sha256": text_hash(artifact_texts[call_id]),
            "accepted_attempt_id": accepted_id,
            "verdict": artifact["verdict"],
            "recommended_repair": artifact["recommended_repair"],
            "observed_service_tier": observed,
            "call_spec_sha256": artifact["call_spec_sha256"],
            "check_plan_skeleton_sha256": artifact["check_plan_skeleton_sha256"],
            "call_config_sha256": artifact["call_config_sha256"],
            "check_plan_sha256": artifact["check_plan_sha256"],
            "previous_artifact_sha256": [entry["artifact_sha256"] for entry in provenance["accepted_history"]],
        }
    return {
        "schema_version": schema_version(inputs.documents),
        "stage": STAGE,
        "dataset_id": inputs.dataset_id,
        "generator": {"package": "fyp_evaluation.plan_verifications", "version": GENERATOR_VERSION},
        "hashing": HASHING,
        "upstream": {"call_config_record_sha256": inputs.upstream.config_record_sha256,
                     "check_plan_record_sha256": inputs.plan_record_sha256, **inputs.source_hashes()},
        "calls": calls,
        "summary": {"accepted_calls": len(calls), "total_calls": len(inputs.call_ids),
                    "reopened_calls": sum(1 for entry in calls.values() if entry["previous_artifact_sha256"]),
                    "verdicts": verdicts, "tier_not_echoed": tier_not_echoed},
    }
