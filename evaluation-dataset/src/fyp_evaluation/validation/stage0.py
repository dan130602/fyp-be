"""G0 (Stage 0) contracts for one versioned dataset release.

Validates the five Stage 0 documents against their schemas, then checks
cross-file relationships that no single schema can express. Declarative only:
this module allocates nothing, implements no spoken-form normalization and
never chooses or changes a gold answer.
"""

import re
from datetime import date
from pathlib import Path

from jsonschema import Draft202012Validator

from .schemas import load_json, validate

STAGE0_SCHEMA_NAMES = ("check_semantics", "normalization", "edge_case_catalog", "generation_plan")
STAGE0_FILES = {
    "checklist": "checklist.json",
    "check_semantics": "check_semantics.json",
    "normalization": "normalization.json",
    "edge_case_catalog": "edge_case_catalog.json",
    "generation_plan": "authoring/generation_plan.json",
}
DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parents[3] / "schemas"

# Generation-plan count maps that select values for one checklist field.
_COUNT_FIELDS = {
    "transaction_direction_counts": "transaction_direction",
    "order_type_counts": "order_type",
    "transaction_currency_counts": "transaction_currency",
}
_FAIL_TOKENS = {"N/A", "NA", "NOT_APPLICABLE"}
_TERMINAL_STEPS = {"EXACT_MATCH", "NUMERIC_EQUAL"}


def _stage0_schema(schema_dir, name):
    schema = load_json(Path(schema_dir) / "stage0" / f"{name}.schema.json")
    Draft202012Validator.check_schema(schema)
    return schema


def check_stage0_schemas(schema_dir=DEFAULT_SCHEMA_DIR):
    """Check the Stage 0 schema definitions against the Draft 2020-12 metaschema."""
    for name in STAGE0_SCHEMA_NAMES:
        _stage0_schema(schema_dir, name)


def load_stage0(dataset_root):
    """Read the five Stage 0 documents of one release; no validation."""
    root = Path(dataset_root)
    return {name: load_json(root / relative) for name, relative in STAGE0_FILES.items()}


def _structural(documents, schema_dir):
    errors = [f"checklist: {e}" for e in validate(documents["checklist"], "checklist", schema_dir)]
    for name in STAGE0_SCHEMA_NAMES:
        validator = Draft202012Validator(_stage0_schema(schema_dir, name))
        errors += [f"{name}: {e.json_path}: {e.message}" for e in validator.iter_errors(documents[name])]
    return errors


def _no_applicability(documents):
    """All questions apply to every call: reject N/A labels and applicability keys."""
    errors = []

    def walk(name, value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                if re.search(r"applicab", key, re.IGNORECASE):
                    errors.append(f"{name}: {path}.{key}: applicability keys are not allowed")
                walk(name, child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(name, child, f"{path}[{index}]")
        elif isinstance(value, str) and value.strip().upper() in _FAIL_TOKENS:
            errors.append(f"{name}: {path}: N/A-style labels are not allowed")

    for name, document in documents.items():
        walk(name, document, "$")
    return errors


def _check_semantics(checklist, semantics, normalization):
    errors = []
    items = {item["id"]: item for item in checklist}
    entries = semantics["checks"]
    if [entry["question_id"] for entry in entries] != list(items):
        errors.append("check_semantics: checks must list exactly the checklist IDs in checklist order")
    profiles = normalization["profiles"]
    referenced = set()
    for entry in entries:
        item = items.get(entry["question_id"])
        if item is None:
            continue
        is_compliance = item["type"] == "COMPLIANCE"
        if is_compliance != ("pass_condition" in entry):
            kind = "pass/fail conditions" if is_compliance else "an extraction resolution rule"
            errors.append(f"check_semantics: {item['id']}: {item['type']} requires {kind}")
            continue
        if is_compliance:
            continue
        profile_id = entry["normalization_profile"]
        referenced.add(profile_id)
        profile = profiles.get(profile_id)
        if profile is None:
            errors.append(f"check_semantics: {item['id']}: unknown normalization profile {profile_id}")
        elif (item["field"] == "dob") != (profile["value_type"] == "dob_dd_month_yyyy"):
            errors.append(f"check_semantics: {item['id']}: field {item['field']} cannot use value_type {profile['value_type']}")
    if unused := sorted(profiles.keys() - referenced):
        errors.append(f"normalization: profiles not referenced by check_semantics: {unused}")
    rule_ids = [rule["id"] for rule in semantics["cross_check_rules"]]
    if len(rule_ids) != len(set(rule_ids)):
        errors.append("check_semantics: duplicate cross_check_rules id")
    for rule in semantics["cross_check_rules"]:
        for condition in [rule["if"], *rule["then"]]:
            item = items.get(condition["question_id"])
            if item is None or item["type"] != "COMPLIANCE":
                errors.append(f"check_semantics: {rule['id']}: {condition['question_id']} is not a compliance question")
        if any(c["question_id"] == rule["if"]["question_id"] for c in rule["then"]):
            errors.append(f"check_semantics: {rule['id']}: a rule cannot constrain its own condition question")
    return errors


def _normalization(normalization, catalog, semantics):
    errors = []
    cases = {case["id"]: case for case in catalog["edge_cases"]}
    profile_questions = {}
    for entry in semantics["checks"]:
        if "normalization_profile" in entry:
            profile_questions.setdefault(entry["normalization_profile"], set()).add(entry["question_id"])
    enabled = set()
    for profile_id, profile in normalization["profiles"].items():
        where = f"normalization: {profile_id}"
        value_type, canonical = profile["value_type"], profile["canonical"]
        if "pattern" in canonical:
            try:
                re.compile(canonical["pattern"])
            except re.error as error:
                errors.append(f"{where}: invalid canonical pattern: {error}")
        if "allowed_values" in canonical and value_type != "text":
            errors.append(f"{where}: allowed_values require value_type text")
        if ("minimum" in canonical or "exclusive_minimum" in canonical) and value_type not in {"integer", "number"}:
            errors.append(f"{where}: numeric bounds require a numeric value_type")
        if "calendar_valid" in canonical and value_type != "dob_dd_month_yyyy":
            errors.append(f"{where}: calendar_valid requires value_type dob_dd_month_yyyy")
        if "magnitude_words" in profile and value_type != "number":
            errors.append(f"{where}: magnitude_words require value_type number")
        if "unit_words" in profile and value_type not in {"integer", "number"}:
            errors.append(f"{where}: unit_words require a numeric value_type")
        aliases = profile.get("aliases", {})
        if aliases and set(aliases) - set(canonical.get("allowed_values", [])):
            errors.append(f"{where}: alias keys must be canonical allowed_values")
        surface = [form.casefold() for forms in aliases.values() for form in forms]
        if len(surface) != len(set(surface)):
            errors.append(f"{where}: an alias surface form maps to more than one canonical value")
        rejected = {entry["form"].casefold() for entry in profile.get("rejected_surface_forms", [])}
        if rejected & set(surface):
            errors.append(f"{where}: a rejected surface form is also an alias")
        steps = profile["comparison"]["steps"]
        if steps[-1] not in _TERMINAL_STEPS or any(step in _TERMINAL_STEPS for step in steps[:-1]):
            errors.append(f"{where}: comparison must end with exactly one EXACT_MATCH or NUMERIC_EQUAL step")
        if "NUMERIC_EQUAL" in steps and value_type not in {"integer", "number"}:
            errors.append(f"{where}: NUMERIC_EQUAL requires a numeric value_type")
        if "MAP_ALIASES" in steps and not aliases:
            errors.append(f"{where}: MAP_ALIASES requires aliases")
        if "EXPAND_MAGNITUDE_WORDS" in steps and "magnitude_words" not in profile:
            errors.append(f"{where}: EXPAND_MAGNITUDE_WORDS requires magnitude_words")
        if "STRIP_UNIT_WORDS" in steps and "unit_words" not in profile:
            errors.append(f"{where}: STRIP_UNIT_WORDS requires unit_words")
        forms = profile["spoken_forms"]
        form_ids = [form["id"] for form in forms]
        if len(form_ids) != len(set(form_ids)):
            errors.append(f"{where}: duplicate spoken form id")
        defaults = [form["id"] for form in forms if form["usage"] == "DEFAULT"]
        if len(defaults) != 1:
            errors.append(f"{where}: exactly one DEFAULT spoken form is required")
        elif profile["reference_text"]["default_form"] != defaults[0]:
            errors.append(f"{where}: reference_text.default_form must name the DEFAULT spoken form")
        for form in forms:
            case_id = form.get("enabled_by_edge_case")
            if case_id is None:
                continue
            enabled.add(case_id)
            case = cases.get(case_id)
            if case is None:
                errors.append(f"{where}: {form['id']} references unknown edge case {case_id}")
            elif not set(case["allowed_target_question_ids"]) & profile_questions.get(profile_id, set()):
                errors.append(f"{where}: {form['id']}: {case_id} targets no question using this profile")
    for case in catalog["edge_cases"]:
        if case["plan_template"]["evidence_mode"] == "SPOKEN_FORM_NORMALISATION" and case["id"] not in enabled:
            errors.append(f"normalization: no spoken form is enabled by {case['id']}")
    return errors


def _catalog(checklist, catalog, semantics, normalization):
    errors = []
    items = {item["id"]: item for item in checklist}
    profiles = normalization["profiles"]
    profile_of = {e["question_id"]: profiles.get(e["normalization_profile"])
                  for e in semantics["checks"] if "normalization_profile" in e}
    ids = [case["id"] for case in catalog["edge_cases"]]
    if len(ids) != len(set(ids)):
        errors.append("edge_case_catalog: duplicate edge case id")
    cases = {case["id"]: case for case in catalog["edge_cases"]}
    for case in catalog["edge_cases"]:
        where = f"edge_case_catalog: {case['id']}"
        targets = case["allowed_target_question_ids"]
        if unknown := [t for t in targets if t not in items]:
            errors.append(f"{where}: unknown target questions {unknown}")
            continue
        compliance_family = case["family"] == "COMPLIANCE_SEMANTICS"
        if any((items[t]["type"] == "COMPLIANCE") != compliance_family for t in targets):
            expected = "compliance" if compliance_family else "extraction"
            errors.append(f"{where}: {case['family']} must target only {expected} questions")
        for other in case.get("mutually_exclusive_with", []):
            if other == case["id"] or other not in cases:
                errors.append(f"{where}: invalid mutually_exclusive_with entry {other}")
            elif case["id"] not in cases[other].get("mutually_exclusive_with", []):
                errors.append(f"{where}: mutual exclusion with {other} must be symmetric")
        if ("antecedent_speaker" in case["required_parameters"]) != ("allowed_antecedent_speakers" in case):
            errors.append(f"{where}: antecedent_speaker parameter and allowed_antecedent_speakers must appear together")
        if case.get("target_scope") == "MULTI_QUESTION" and "target_question_ids" not in case["required_parameters"]:
            errors.append(f"{where}: MULTI_QUESTION scope requires a target_question_ids parameter")
        requirements = case.get("verdict_requirements", {})
        for question_id in requirements:
            if items.get(question_id, {}).get("type") != "COMPLIANCE":
                errors.append(f"{where}: verdict requirement on non-compliance question {question_id}")
        related = case["plan_template"].get("required_related_gold")
        if related is not None and related != requirements:
            errors.append(f"{where}: plan_template.required_related_gold must equal verdict_requirements")
        if "target_verdict_requirement" in case and not compliance_family:
            errors.append(f"{where}: target_verdict_requirement applies only to compliance targets")
        for rule in semantics["cross_check_rules"]:
            condition = rule["if"]
            if requirements.get(condition["question_id"]) != condition["verdict"]:
                continue
            for implied in rule["then"]:
                actual = requirements.get(implied["question_id"])
                if actual is not None and actual != implied["verdict"]:
                    errors.append(f"{where}: verdict_requirements contradict {rule['id']}")
        prerequisite = case.get("gold_prerequisites")
        if prerequisite is not None:
            numeric = prerequisite["kind"].startswith("NUMERIC_")
            for target in targets:
                profile = profile_of.get(target)
                value_type = profile and profile["value_type"]
                if numeric != (value_type in {"integer", "number"}):
                    errors.append(f"{where}: {prerequisite['kind']} cannot apply to {target} ({value_type})")
    return errors


def _coherent_pairs(extraction, currency):
    bounds = extraction["amount_quantity_coherence"]["ranges"].get(currency)
    if bounds is None:
        return []
    return [
        (amount, quantity)
        for amount in extraction["transaction_amount_values_by_currency"].get(currency, [])
        for quantity in extraction["transaction_quantity_values"]
        if bounds["min"] <= amount / quantity <= bounds["max"]
    ]


def _meets(prerequisite, value):
    if prerequisite["kind"] == "NUMERIC_MINIMUM":
        return value >= prerequisite["minimum"]
    if prerequisite["kind"] == "NUMERIC_MINIMUM_AND_MULTIPLE":
        return value >= prerequisite["minimum"] and value % prerequisite["multiple_of"] == 0
    return True


def _generation_plan(checklist, plan, catalog, semantics, normalization):
    errors = []
    where = "generation_plan"
    total = plan["num_calls"]
    fields = {item["field"]: item["id"] for item in checklist if item["type"] != "COMPLIANCE"}
    profile_of = {e["question_id"]: normalization["profiles"].get(e["normalization_profile"])
                  for e in semantics["checks"] if "normalization_profile" in e}

    def require_total(label, counts):
        if sum(counts) != total:
            errors.append(f"{where}: {label} counts sum to {sum(counts)}, not num_calls={total}")

    require_total("language", plan["language_counts"].values())
    durations = plan["duration_profiles"]
    require_total("duration", [d["count"] for d in durations])
    if len({d["profile"] for d in durations}) != len(durations):
        errors.append(f"{where}: duplicate duration profile")
    for duration in durations:
        if duration["min_seconds"] > duration["max_seconds"]:
            errors.append(f"{where}: {duration['profile']} min_seconds exceeds max_seconds")

    vocabulary = plan["call_intent_vocabulary"]
    intent_sets = plan["call_intent_set_counts"]
    require_total("call intent set", [s["count"] for s in intent_sets])
    seen_sets = set()
    for entry in intent_sets:
        intents = entry["call_intents"]
        if unknown := [i for i in intents if i not in vocabulary]:
            errors.append(f"{where}: call intents {unknown} are not in call_intent_vocabulary")
        elif intents != sorted(intents, key=vocabulary.index):
            errors.append(f"{where}: call_intents {intents} must follow vocabulary order")
        if tuple(intents) in seen_sets:
            errors.append(f"{where}: duplicate call intent set {intents}")
        seen_sets.add(tuple(intents))

    composition = plan["call_composition_counts"]
    require_total("call composition", composition.values())
    some_fail = composition["some_fail_without_edge_cases"] + composition["some_fail_with_edge_cases"]
    with_edges = composition["all_pass_with_edge_cases"] + composition["some_fail_with_edge_cases"]
    failures = plan["compliance_distribution"]["failures_per_non_all_pass_call"]
    per_call = plan["edge_case_distribution"]["edge_cases_per_non_clean_call"]
    for label, bounds in (("failures_per_non_all_pass_call", failures), ("edge_cases_per_non_clean_call", per_call)):
        if bounds["min"] > bounds["max"]:
            errors.append(f"{where}: {label} min exceeds max")
    compliance_count = sum(item["type"] == "COMPLIANCE" for item in checklist)
    needed = plan["compliance_distribution"]["min_fail_cases_per_check"] * compliance_count
    if needed > some_fail * failures["max"]:
        errors.append(f"{where}: {needed} required FAIL cells exceed capacity {some_fail * failures['max']}")

    case_ids = {case["id"] for case in catalog["edge_cases"]}
    if unknown := sorted(set(plan["planning_rules"]["per_question_limit_exempt_edge_cases"]) - case_ids):
        errors.append(f"{where}: per_question_limit_exempt_edge_cases references unknown edge cases {unknown}")
    quotas = plan["edge_case_distribution"]["family_quotas"]
    families = {case["family"] for case in catalog["edge_cases"]}
    if set(quotas) != families:
        errors.append(f"{where}: family_quotas keys must equal catalog families {sorted(families)}")
    quota_total = sum(quotas.values())
    if not with_edges * per_call["min"] <= quota_total <= with_edges * per_call["max"]:
        errors.append(f"{where}: {quota_total} edge-case assignments cannot fill {with_edges} calls "
                      f"at {per_call['min']}-{per_call['max']} each")
    forces_fail = ["FAIL" in case.get("verdict_requirements", {}).values() for case in catalog["edge_cases"]]
    if composition["all_pass_with_edge_cases"] and all(forces_fail):
        errors.append(f"{where}: all-PASS calls with edge cases need an edge case that forces no FAIL")

    extraction = plan["extraction_distribution"]
    for key, field in _COUNT_FIELDS.items():
        counts = extraction[key]
        require_total(key, counts.values())
        profile = profile_of.get(fields.get(field))
        allowed = profile and profile["canonical"].get("allowed_values")
        if allowed is None:
            errors.append(f"{where}: {key}: field {field} has no profile with allowed_values")
        elif set(counts) != set(allowed):
            errors.append(f"{where}: {key} keys must equal normalization allowed_values {allowed}")
    currencies = set(extraction["transaction_currency_counts"])
    names = [product["name"] for product in extraction["product_pool"]]
    if len(names) != len(set(names)):
        errors.append(f"{where}: duplicate product name")
    for product in extraction["product_pool"]:
        if extra := set(product["allowed_currencies"]) - currencies:
            errors.append(f"{where}: {product['name']} allows unplanned currencies {sorted(extra)}")
    if set(extraction["transaction_amount_values_by_currency"]) != currencies:
        errors.append(f"{where}: transaction_amount_values_by_currency keys must equal transaction currencies")
    if set(extraction["amount_quantity_coherence"]["ranges"]) != currencies:
        errors.append(f"{where}: amount_quantity_coherence ranges must cover exactly the transaction currencies")
    for currency, count in extraction["transaction_currency_counts"].items():
        if not count:
            continue
        if not any(currency in p["allowed_currencies"] for p in extraction["product_pool"]):
            errors.append(f"{where}: no product allows planned currency {currency}")
        bounds = extraction["amount_quantity_coherence"]["ranges"].get(currency)
        if bounds and bounds["min"] > bounds["max"]:
            errors.append(f"{where}: {currency} unit-price range min exceeds max")
        elif not _coherent_pairs(extraction, currency):
            errors.append(f"{where}: {currency} has no coherent amount/quantity pair")

    for case in catalog["edge_cases"]:
        prerequisite = case.get("gold_prerequisites")
        if prerequisite is None or not prerequisite["kind"].startswith("NUMERIC_"):
            continue
        for target in case["allowed_target_question_ids"]:
            field = next((f for f, q in fields.items() if q == target), None)
            pairs = [pair for currency in currencies for pair in _coherent_pairs(extraction, currency)]
            index = {"transaction_amount": 0, "transaction_quantity": 1}.get(field)
            if index is not None and not any(_meets(prerequisite, pair[index]) for pair in pairs):
                errors.append(f"{where}: no coherent {field} value satisfies {case['id']} prerequisites")

    kyc = extraction["synthetic_kyc"]
    identifier = profile_of.get(fields.get("client_identification_number"))
    pattern = identifier and identifier["canonical"].get("pattern")
    nric = kyc["nric_fin"]
    if pattern:
        for prefix in nric["prefixes"]:
            sample = prefix + "0" * nric["digit_count"] + nric["suffix_letters"][0]
            if not re.fullmatch(pattern, sample):
                errors.append(f"{where}: NRIC/FIN prefix {prefix} does not fit the canonical pattern")
    if len(set(nric["suffix_letters"])) != len(nric["suffix_letters"]):
        errors.append(f"{where}: suffix_letters must not repeat")
    try:
        first, last = date.fromisoformat(kyc["dob"]["min_date"]), date.fromisoformat(kyc["dob"]["max_date"])
    except ValueError as error:
        errors.append(f"{where}: invalid DOB range date: {error}")
    else:
        if (last - first).days + 1 < total:
            errors.append(f"{where}: DOB range cannot provide {total} unique dates")
    return errors


def validate_stage0(documents, schema_dir=DEFAULT_SCHEMA_DIR):
    """Return errors for already-loaded Stage 0 documents; never mutate them."""
    errors = _structural(documents, schema_dir)
    if errors:
        return errors
    errors = _no_applicability(documents)
    checklist, semantics = documents["checklist"], documents["check_semantics"]
    normalization, catalog = documents["normalization"], documents["edge_case_catalog"]
    errors += _check_semantics(checklist, semantics, normalization)
    errors += _normalization(normalization, catalog, semantics)
    errors += _catalog(checklist, catalog, semantics, normalization)
    errors += _generation_plan(checklist, documents["generation_plan"], catalog, semantics, normalization)
    return errors


def validate_stage0_release(dataset_root, schema_dir=DEFAULT_SCHEMA_DIR):
    """Load and validate one release's Stage 0 documents and its dataset identity."""
    root = Path(dataset_root)
    documents = load_stage0(root)
    errors = validate_stage0(documents, schema_dir)
    manifest = load_json(root / "dataset_manifest.json")
    if documents["normalization"].get("dataset_id") != manifest.get("dataset_id"):
        errors.append("normalization: dataset_id must equal dataset_manifest.json dataset_id")
    return errors
