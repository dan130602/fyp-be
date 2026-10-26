"""Python-owned compilation of a G7 report: identity, upstream hashes, expanded evidence, derived verdict and quality review.

The model never states a verdict. Precedence over check rows, edge-case rows and issues: any MISMATCH -> INVALID; otherwise
any UNCLEAR -> UNCLEAR; otherwise VALID. A whole-dialogue issue therefore decides the verdict even when all 20 checks MATCH.
`quality_review` is REVIEW_REQUIRED when any quality finding is SERIOUS and never alters the verdict. The recommended repair
is derived from the issues' advisory scopes and is never executed here. Compilation is a pure function of (inputs bound to the
verified G3/G4/G5 versions and the verified G6 dialogue, accepted parsed response, recorded generation metadata), so `--check`
rebuilds every report without any provider envelope.
"""

from fyp_evaluation import register_guide
from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json, text_hash
from fyp_evaluation.llm.canonical import canonical_response

from .config import GENERATOR_VERSION, HASHING, QUALITY_REVIEWS, STAGE, VERDICTS, schema_version
from .validate_response import expand_all


def derive_verdict(checks, edge_cases, issues, language_checks=()):
    assessments = ([r["assessment"] for r in checks] + [r["assessment"] for r in edge_cases] + [r["assessment"] for r in language_checks]
                   + [i["assessment"] for i in issues])
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
    if "UPSTREAM_REVIEW" in scopes:
        return "UPSTREAM_REVIEW"
    if "REVIEW" in scopes:
        return "REVIEW"
    return "G6_ONLY"


def derive_quality_review(quality_findings):
    return "REVIEW_REQUIRED" if any(f["severity"] == "SERIOUS" for f in quality_findings) else "CLEAR"


def generation_block(generation, parsed, response_schema):
    return {
        "attempt_id": generation["attempt_id"], "model": generation["model"], "params": dict(generation["params"]),
        "prompt_template_sha256": generation["prompt_template_sha256"],
        "response_schema_sha256": generation["response_schema_sha256"],
        "accepted_response_sha256": text_hash(serialize_json(canonical_response(parsed, response_schema))),
    }


def _counts(rows):
    return {"match": sum(1 for r in rows if r["assessment"] == "MATCH"), "mismatch": sum(1 for r in rows if r["assessment"] == "MISMATCH"),
            "unclear": sum(1 for r in rows if r["assessment"] == "UNCLEAR")}


def compile_artifact(inputs, ctx, parsed, generation, response_schema):
    checks, edge_cases, issues, quality, language, errors = expand_all(parsed, ctx)
    if errors:
        raise ValueError("cannot compile the dialogue verification: " + "; ".join(errors[:5]))
    multilingual = ctx.multilingual
    verdict = derive_verdict(checks, edge_cases, issues, language)
    by_code = {}
    for issue in issues:
        by_code[issue["code"]] = by_code.get(issue["code"], 0) + 1
    call_id = ctx.call_id
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
        "dialogue_sha256": inputs.dialogue_hashes[call_id],
        "source": {**inputs.source_hashes(), **(register_guide.recorded(generation) if multilingual else {})},
        "generation": generation_block(generation, parsed, response_schema),
        "verdict": verdict,
        "recommended_repair": derive_repair(verdict, issues),
        "quality_review": derive_quality_review(quality),
        "summary": {"checks": _counts(checks), "edge_cases": _counts(edge_cases), "issues_by_code": dict(sorted(by_code.items())),
                    "quality_by_severity": {"NOTE": sum(1 for f in quality if f["severity"] == "NOTE"),
                                            "SERIOUS": sum(1 for f in quality if f["severity"] == "SERIOUS")}},
        "checks": checks,
        "edge_cases": edge_cases,
        "issues": issues,
        "quality_findings": quality,
    }
    if multilingual:
        artifact["summary"]["language_checks"] = _counts(language)
        artifact["language_checks"] = language
    return artifact


def build_record(inputs, artifact_texts, provenance_by_call):
    """Deterministic record over the accepted reports (no timestamps)."""
    calls, verdicts, reviews, tier_not_echoed = {}, dict.fromkeys(VERDICTS, 0), dict.fromkeys(QUALITY_REVIEWS, 0), 0
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
        reviews[artifact["quality_review"]] += 1
        calls[call_id] = {
            "artifact_sha256": text_hash(artifact_texts[call_id]),
            "accepted_attempt_id": accepted_id,
            "verdict": artifact["verdict"],
            "quality_review": artifact["quality_review"],
            "recommended_repair": artifact["recommended_repair"],
            "observed_service_tier": observed,
            "call_spec_sha256": artifact["call_spec_sha256"],
            "check_plan_skeleton_sha256": artifact["check_plan_skeleton_sha256"],
            "call_config_sha256": artifact["call_config_sha256"],
            "check_plan_sha256": artifact["check_plan_sha256"],
            "plan_verification_sha256": artifact["plan_verification_sha256"],
            "dialogue_sha256": artifact["dialogue_sha256"],
            "previous_artifact_sha256": [entry["artifact_sha256"] for entry in provenance["accepted_history"]],
        }
    return {
        "schema_version": schema_version(inputs.documents),
        "stage": STAGE,
        "dataset_id": inputs.dataset_id,
        "generator": {"package": "fyp_evaluation.dialogue_verifications", "version": GENERATOR_VERSION},
        "hashing": HASHING,
        "upstream": {"dialogue_record_sha256": inputs.g6_record_sha256, **inputs.source_hashes()},
        "calls": calls,
        "summary": {"accepted_calls": len(calls), "total_calls": len(inputs.call_ids),
                    "reopened_calls": sum(1 for entry in calls.values() if entry["previous_artifact_sha256"]),
                    "verdicts": verdicts, "quality_reviews": reviews, "tier_not_echoed": tier_not_echoed},
    }
