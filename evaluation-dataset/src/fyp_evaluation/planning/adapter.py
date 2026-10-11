"""Minimal adapter from release-local normalization to the legacy rule shape.

The existing matrix tooling (`check_matrix`, `load_matrix`, `compile_expected`,
`validate_checklist`) expects `{rules, compliance_rule, field_rules}`. This
builds that shape in memory from the release's profiles and check semantics.
It writes nothing and never consults configs/normalization.json.
"""

LEGACY_NORMALIZATION_VERSION = "0.3.0"
COMPLIANCE_RULE = "exact_compliance"


def legacy_normalization(checklist, check_semantics, normalization):
    """Return a legacy-shaped normalization mapping each extraction field to its profile."""
    profile_of = {
        entry["question_id"]: entry["normalization_profile"]
        for entry in check_semantics["checks"] if "normalization_profile" in entry
    }
    rules = {COMPLIANCE_RULE: {"operation": "exact", "value_type": "verdict"}}
    field_rules = {}
    for item in checklist:
        if item["type"] == "COMPLIANCE":
            continue
        profile_id = profile_of[item["id"]]
        rules[profile_id] = {
            "operation": "exact",
            "value_type": normalization["profiles"][profile_id]["value_type"],
        }
        field_rules[item["field"]] = profile_id
    return {
        "schema_version": LEGACY_NORMALIZATION_VERSION,
        "rules": rules,
        "compliance_rule": COMPLIANCE_RULE,
        "field_rules": field_rules,
    }
