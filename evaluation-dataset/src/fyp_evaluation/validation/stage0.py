"""G0 (Stage 0) contracts for one versioned dataset release.

Validates the Stage 0 documents against their schemas, then checks
cross-file relationships that no single schema can express. Declarative only:
this module allocates nothing, implements no spoken-form normalization and
never chooses or changes a gold answer.

A release is either monolingual (generation plan 1.0: the five original
contracts) or multilingual (generation plan 2.0: the same five plus
language_profiles.json, with catalog and normalization at schema 2.0).
"""

import re
from datetime import date
from pathlib import Path

from jsonschema import Draft202012Validator

from .schemas import load_json, validate

STAGE0_SCHEMA_NAMES = ("check_semantics", "normalization", "edge_case_catalog", "generation_plan")
MULTILINGUAL_SCHEMA_NAMES = ("generation_plan.v2", "language_profiles")
STAGE0_FILES = {
    "checklist": "checklist.json",
    "check_semantics": "check_semantics.json",
    "normalization": "normalization.json",
    "edge_case_catalog": "edge_case_catalog.json",
    "generation_plan": "authoring/generation_plan.json",
}
LANGUAGE_PROFILES_FILE = "language_profiles.json"
MULTILINGUAL_PLAN_VERSION = "2.0"
DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parents[3] / "schemas"
HAN = re.compile(r"[〇㐀-䶿一-鿿豈-﫿]")
LATIN = re.compile(r"[A-Za-z]")


def is_multilingual(documents):
    """True for a release whose generation plan is the multilingual 2.0 contract."""
    return documents["generation_plan"].get("schema_version") == MULTILINGUAL_PLAN_VERSION


def stage0_files(documents):
    """{document name: relative path} of every G0 file of this release (hashed by G1)."""
    files = dict(STAGE0_FILES)
    if "language_profiles" in documents:
        files["language_profiles"] = LANGUAGE_PROFILES_FILE
    return files

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
    for name in STAGE0_SCHEMA_NAMES + MULTILINGUAL_SCHEMA_NAMES:
        _stage0_schema(schema_dir, name)


def load_stage0(dataset_root):
    """Read the Stage 0 documents of one release (language_profiles.json when present); no validation."""
    root = Path(dataset_root)
    documents = {name: load_json(root / relative) for name, relative in STAGE0_FILES.items()}
    if (root / LANGUAGE_PROFILES_FILE).is_file():
        documents["language_profiles"] = load_json(root / LANGUAGE_PROFILES_FILE)
    return documents


def _schema_name(name, document):
    if name == "generation_plan" and isinstance(document, dict) and document.get("schema_version") == MULTILINGUAL_PLAN_VERSION:
        return "generation_plan.v2"
    return name


def _structural(documents, schema_dir):
    errors = [f"checklist: {e}" for e in validate(documents["checklist"], "checklist", schema_dir)]
    names = STAGE0_SCHEMA_NAMES + (("language_profiles",) if "language_profiles" in documents else ())
    for name in names:
        validator = Draft202012Validator(_stage0_schema(schema_dir, _schema_name(name, documents[name])))
        errors += [f"{name}: {e.json_path}: {e.message}" for e in validator.iter_errors(documents[name])]
    return errors


def _versions(documents):
    """A release is wholly 1.0 or wholly multilingual; mixing contract generations is refused."""
    multilingual = is_multilingual(documents)
    expected = "2.0" if multilingual else "1.0"
    errors = [f"{name}: schema_version must be {expected} in a {'multilingual' if multilingual else 'monolingual'} release"
              for name in ("normalization", "edge_case_catalog") if documents[name]["schema_version"] != expected]
    if multilingual != ("language_profiles" in documents):
        errors.append(f"{LANGUAGE_PROFILES_FILE} is required exactly when generation_plan.json is schema 2.0")
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
            scope = entry["speaker_scope"]
            if entry["presence_pattern"] == "EMPLOYEE_ONLY" and scope != ["EMPLOYEE"]:
                errors.append(f"check_semantics: {item['id']}: EMPLOYEE_ONLY requires speaker_scope [EMPLOYEE]")
            if entry["presence_pattern"] in {"EXCHANGE", "ANY_SPEAKER"} and set(scope) != {"EMPLOYEE", "CLIENT"}:
                errors.append(f"check_semantics: {item['id']}: {entry['presence_pattern']} requires both speakers in speaker_scope")
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
    ordering_ids = [rule["id"] for rule in semantics["ordering_rules"]]
    if len(ordering_ids) != len(set(ordering_ids)):
        errors.append("check_semantics: duplicate ordering_rules id")
    graph = {}
    for rule in semantics["ordering_rules"]:
        subject = items.get(rule["question_id"])
        if subject is None or subject["type"] != "COMPLIANCE":
            errors.append(f"check_semantics: {rule['id']}: {rule['question_id']} is not a compliance question")
        after = rule["after"]
        if after["type"] == "CHECK_EVIDENCE":
            referenced = [after["question_id"]]
            if any(items.get(q, {}).get("type") != "COMPLIANCE" for q in referenced):
                errors.append(f"check_semantics: {rule['id']}: CHECK_EVIDENCE must name a compliance question")
        else:
            referenced = after["question_ids"]
            if any(items.get(q, {}).get("type") != "TRANSACTION" for q in referenced):
                errors.append(f"check_semantics: {rule['id']}: FINAL_STATE must name transaction questions")
        if rule["question_id"] in referenced:
            errors.append(f"check_semantics: {rule['id']}: a rule cannot order a question after itself")
        graph.setdefault(rule["question_id"], set()).update(q for q in referenced if q in items)
    for start_node in graph:
        seen, stack = set(), [start_node]
        while stack:
            for nxt in graph.get(stack.pop(), ()):
                if nxt == start_node:
                    errors.append(f"check_semantics: ordering_rules form a cycle through {start_node}")
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
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


def _multilingual_profile(where, profile, languages):
    """Schema 2.0 profile rules: a language on every form, one DEFAULT per language, consistent generation surfaces."""
    errors = []
    forms = {form["id"]: form for form in profile["spoken_forms"]}
    if missing := sorted(i for i, form in forms.items() if form.get("language") not in languages):
        errors.append(f"{where}: spoken forms {missing} need a language from normalization.languages")
        return errors
    for language in languages:
        defaults = [i for i, form in forms.items() if form["usage"] == "DEFAULT" and form["language"] == language]
        if len(defaults) != 1:
            errors.append(f"{where}: exactly one DEFAULT spoken form is required for language {language}")
        elif language == languages[0] and profile["reference_text"]["default_form"] != defaults[0]:
            errors.append(f"{where}: reference_text.default_form must name the {language} DEFAULT spoken form")
    for form in forms.values():
        remaining = forms.get(form.get("remaining_characters_form"))
        if remaining is not None and remaining["language"] != form["language"]:
            errors.append(f"{where}: {form['id']}: remaining_characters_form must be in the same language")
    for entry in profile.get("rejected_surface_forms", []):
        if entry.get("language", languages[0]) not in languages:
            errors.append(f"{where}: rejected form {entry['form']!r} has an unknown language")
    aliases = profile.get("aliases", {})
    surfaces = profile.get("generation_surfaces")
    if surfaces is not None:
        allowed = profile["canonical"].get("allowed_values", [])
        if list(surfaces) != allowed:
            errors.append(f"{where}: generation_surfaces must list every allowed value in canonical order")
        for value, by_form in surfaces.items():
            for form_id, texts in by_form.items():
                form = forms.get(form_id)
                if form is None or form["usage"] == "EDGE_CASE_ONLY":
                    errors.append(f"{where}: {value}: generation surface form {form_id} must be a DEFAULT or PERMITTED spoken form")
                    continue
                for text in texts:
                    if text not in aliases.get(value, []):
                        errors.append(f"{where}: {value}: generation surface {text!r} must also be an accepted alias")
                    if bool(HAN.search(text)) != (form["language"] == "zh") or (form["language"] == "zh" and LATIN.search(text)):
                        errors.append(f"{where}: {value}: surface {text!r} does not match the script of {form['language']} form {form_id}")
            defaults = {f["id"] for f in forms.values() if f["usage"] == "DEFAULT"}
            if missing := sorted(defaults - set(by_form)):
                errors.append(f"{where}: {value}: no generation surface for DEFAULT forms {missing}")
    names = profile.get("language_names")
    if ("MAP_LANGUAGE_NAMES" in profile["comparison"]["steps"]) != (names is not None):
        errors.append(f"{where}: MAP_LANGUAGE_NAMES and language_names must appear together")
    if names is not None:
        if "values_source" not in profile["canonical"]:
            errors.append(f"{where}: language_names apply only to a profile whose values come from a values_source")
        seen = {}
        for canonical, by_language in names.items():
            for language, entry in by_language.items():
                if language not in languages or language == languages[0]:
                    errors.append(f"{where}: {canonical}: language_names language {language} must be a non-primary release language")
                    continue
                if language == "zh" and (not HAN.search(entry["surface"]) or LATIN.search(entry["surface"])
                                         or re.search(r"[0-9]", entry["surface"])):
                    errors.append(f"{where}: {canonical}: zh name {entry['surface']!r} must be Han script with no Latin letters or Arabic digits")
                if entry["surface"] in seen.setdefault(language, set()):
                    errors.append(f"{where}: {language} name {entry['surface']!r} is used by more than one product")
                seen[language].add(entry["surface"])
    elif any(form["language"] != languages[0] for form in forms.values()) and "values_source" in profile["canonical"]:
        errors.append(f"{where}: a non-primary-language product form needs language_names")
    return errors


def _monolingual_profile(where, profile):
    """Schema 1.0 profiles carry none of the multilingual fields."""
    present = [key for key in ("generation_surfaces", "language_names") if key in profile]
    present += [f"spoken_forms[{f['id']}].language" for f in profile["spoken_forms"] if "language" in f]
    present += [f"rejected_surface_forms[{e['form']}].{k}" for e in profile.get("rejected_surface_forms", [])
                for k in ("language", "match") if k in e]
    if "MAP_LANGUAGE_NAMES" in profile["comparison"]["steps"]:
        present.append("comparison.MAP_LANGUAGE_NAMES")
    return [f"{where}: {key} is a schema 2.0 field" for key in present]


def _normalization(normalization, catalog, semantics):
    errors = []
    multilingual = normalization["schema_version"] == "2.0"
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
        excluded = canonical.get("excluded_characters", [])
        if excluded and "pattern" in canonical:
            probe = re.compile(canonical["pattern"])
            for character in excluded:
                if any(probe.fullmatch(prefix + "0" * 7 + character) for prefix in "STFGM"):
                    errors.append(f"{where}: canonical pattern still accepts excluded character {character}")
        if ("excluded_characters" in canonical) != ("excluded_characters_reason" in canonical):
            errors.append(f"{where}: excluded_characters and excluded_characters_reason must appear together")
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
        if multilingual:
            errors += _multilingual_profile(where, profile, normalization["languages"])
        else:
            errors += _monolingual_profile(where, profile)
            defaults = [form["id"] for form in forms if form["usage"] == "DEFAULT"]
            if len(defaults) != 1:
                errors.append(f"{where}: exactly one DEFAULT spoken form is required")
            elif profile["reference_text"]["default_form"] != defaults[0]:
                errors.append(f"{where}: reference_text.default_form must name the DEFAULT spoken form")
        for form in forms:
            remaining = form.get("remaining_characters_form")
            if remaining is not None:
                target = next((f for f in forms if f["id"] == remaining), None)
                if target is None or target["usage"] != "DEFAULT":
                    errors.append(f"{where}: {form['id']}: remaining_characters_form must name the DEFAULT spoken form")
                if form.get("applies_to") == "WHOLE_VALUE":
                    errors.append(f"{where}: {form['id']}: WHOLE_VALUE leaves no remaining characters")
            elif form.get("applies_to") in {"ALL_ZERO_DIGITS", "ALL_RUNS_OF_ASSIGNED_VARIANT_LENGTH"}:
                errors.append(f"{where}: {form['id']}: partial coverage requires remaining_characters_form")
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
        if "superseded_values" in case["required_parameters"]:
            for target in targets:
                allowed = (profile_of.get(target) or {}).get("canonical", {}).get("allowed_values")
                if allowed is not None and len(allowed) < 3:
                    errors.append(f"{where}: {target} has only {len(allowed)} allowed values; "
                                  "two superseded values plus the final value need at least three")
        template = case["plan_template"]
        if ("allowed_distractor_speakers" in case) != (template["evidence_mode"] == "DISTRACTOR_PLUS_TARGET"):
            errors.append(f"{where}: allowed_distractor_speakers must appear exactly on DISTRACTOR_PLUS_TARGET cases")
        if template.get("fragment_must_be_incomplete") and template.get("same_turn") is not True:
            errors.append(f"{where}: an incomplete false-start fragment requires same_turn true")
        if ("min_readback_fields" in template) != ("readback_wrong_fields" in template):
            errors.append(f"{where}: min_readback_fields and readback_wrong_fields must appear together")
        if "min_readback_fields" in template and (
                template["min_readback_fields"] > len(targets) or any(items[q]["type"] != "TRANSACTION" for q in targets)):
            errors.append(f"{where}: a readback needs at least min_readback_fields transaction questions to draw from")
        if "wording_constraint" in template and not compliance_family:
            errors.append(f"{where}: wording_constraint applies only to compliance cases")
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


def numeric_prerequisite_met(prerequisite, value):
    """True when a numeric gold value satisfies one numeric gold prerequisite."""
    kind = prerequisite["kind"]
    if kind == "NUMERIC_MINIMUM":
        return value >= prerequisite["minimum"]
    if kind == "NUMERIC_MINIMUM_AND_MULTIPLE":
        return value >= prerequisite["minimum"] and value % prerequisite["multiple_of"] == 0
    if kind == "NUMERIC_MINIMUM_NOT_MULTIPLE":
        return value >= prerequisite["minimum"] and value % prerequisite["not_multiple_of"] != 0
    raise ValueError(f"Not a numeric gold prerequisite: {kind}")


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
    all_pass = composition["all_pass_without_edge_cases"] + composition["all_pass_with_edge_cases"]
    without_edges = composition["all_pass_without_edge_cases"] + composition["some_fail_without_edge_cases"]
    fail_counts = {int(k): v for k, v in plan["compliance_distribution"]["fail_count_per_call_counts"].items()}
    require_total("fail_count_per_call", fail_counts.values())
    if fail_counts.get(0, 0) != all_pass:
        errors.append(f"{where}: calls with 0 FAILs ({fail_counts.get(0, 0)}) must equal all-PASS calls ({all_pass})")
    compliance_count = sum(item["type"] == "COMPLIANCE" for item in checklist)
    if fail_counts and max(fail_counts) > compliance_count:
        errors.append(f"{where}: a call cannot have more FAILs than the {compliance_count} compliance questions")
    total_fails = sum(number * count for number, count in fail_counts.items())
    needed = plan["compliance_distribution"]["min_fail_cases_per_check"] * compliance_count
    if needed > total_fails:
        errors.append(f"{where}: {needed} required FAIL cells exceed the {total_fails} planned FAIL cells")
    edge_counts = {int(k): v for k, v in plan["edge_case_distribution"]["edge_cases_per_call_counts"].items()}
    require_total("edge_cases_per_call", edge_counts.values())
    if edge_counts.get(0, 0) != without_edges:
        errors.append(f"{where}: calls with 0 edge cases ({edge_counts.get(0, 0)}) must equal calls without edge cases ({without_edges})")

    case_ids = {case["id"] for case in catalog["edge_cases"]}
    if unknown := sorted(set(plan["planning_rules"]["per_question_limit_exempt_edge_cases"]) - case_ids):
        errors.append(f"{where}: per_question_limit_exempt_edge_cases references unknown edge cases {unknown}")
    quotas = plan["edge_case_distribution"]["family_quotas"]
    families = {case["family"] for case in catalog["edge_cases"]}
    if set(quotas) != families:
        errors.append(f"{where}: family_quotas keys must equal catalog families {sorted(families)}")
    quota_total = sum(quotas.values())
    planned_edge_cases = sum(number * count for number, count in edge_counts.items())
    if quota_total != planned_edge_cases:
        errors.append(f"{where}: family_quotas total {quota_total} must equal per-call edge-case total {planned_edge_cases}")
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
            if index is not None and not any(numeric_prerequisite_met(prerequisite, pair[index]) for pair in pairs):
                errors.append(f"{where}: no coherent {field} value satisfies {case['id']} prerequisites")

    kyc = extraction["synthetic_kyc"]
    identifier = profile_of.get(fields.get("client_identification_number"))
    pattern = identifier and identifier["canonical"].get("pattern")
    nric = kyc["nric_fin"]
    if pattern:
        for prefix in nric["prefixes"]:
            for suffix in nric["suffix_letters"]:
                if not re.fullmatch(pattern, prefix + "0" * nric["digit_count"] + suffix):
                    errors.append(f"{where}: NRIC/FIN prefix {prefix} with suffix {suffix} does not fit the canonical pattern")
                    break
    excluded = (identifier or {}).get("canonical", {}).get("excluded_characters", [])
    if clash := sorted(set(excluded) & (set(nric["suffix_letters"]) | set(nric["prefixes"]))):
        errors.append(f"{where}: NRIC/FIN pools contain excluded characters {clash}")
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


def _catalog_languages(catalog, language_doc, normalization):
    """Schema 2.0: every case declares its eligible language profiles and its controlled language variants."""
    errors = []
    profiles = language_doc["profiles"]
    for case in catalog["edge_cases"]:
        where = f"edge_case_catalog: {case['id']}"
        if "language_profiles" not in case or "language_variants" not in case:
            errors.append(f"{where}: language_profiles and language_variants are required in schema 2.0")
            continue
        eligible = case["language_profiles"]
        if unknown := sorted(set(eligible) - set(profiles)):
            errors.append(f"{where}: unknown language profiles {unknown}")
        ids = [variant["variant_id"] for variant in case["language_variants"]]
        if len(ids) != len(set(ids)):
            errors.append(f"{where}: duplicate language variant id")
        for variant in case["language_variants"]:
            label = f"{where}: {variant['variant_id']}"
            if unknown := sorted(set(variant["profiles"]) - set(profiles)):
                errors.append(f"{label}: unknown language profiles {unknown}")
                continue
            if variant["status"] == "EXCLUDED_V1":
                continue
            if outside := sorted(set(variant["profiles"]) - set(eligible)):
                errors.append(f"{label}: profiles {outside} are not eligible for the case")
            for language in variant.get("event_languages", {}).values():
                if language not in normalization["languages"]:
                    errors.append(f"{label}: unknown event language {language}")
                for profile in variant["profiles"]:
                    if language not in profiles.get(profile, {}).get("value_surface_languages", []):
                        errors.append(f"{label}: {profile} does not speak values in {language}")
            if "constraint" in variant and "distractor_values" not in case["required_parameters"]:
                errors.append(f"{label}: a distractor magnitude constraint needs a distractor_values case")
    return errors


def _monolingual_catalog(catalog):
    return [f"edge_case_catalog: {case['id']}: {key} is a schema 2.0 field"
            for case in catalog["edge_cases"] for key in ("language_profiles", "language_variants") if key in case]


def _language_profiles(language_doc, normalization):
    errors = []
    where = "language_profiles"
    languages = list(language_doc["languages"])
    if language_doc["dataset_id"] != normalization["dataset_id"]:
        errors.append(f"{where}: dataset_id must equal normalization.json dataset_id")
    if set(languages) != set(normalization["languages"]):
        errors.append(f"{where}: languages must equal normalization.json languages {normalization['languages']}")
    for name, profile in language_doc["profiles"].items():
        label = f"{where}: {name}"
        values = profile["value_surface_languages"]
        spoken = set()
        if unknown := sorted(set(values) - set(languages)):
            errors.append(f"{label}: unknown value_surface_languages {unknown}")
        if outside := sorted(set(profile["product_name_languages"]) - set(values)):
            errors.append(f"{label}: product_name_languages {outside} must be value_surface_languages")
        for speaker, rules in profile["speakers"].items():
            spoken |= set(rules["spoken_languages"])
            if unknown := sorted(set(rules["spoken_languages"]) - set(languages)):
                errors.append(f"{label}: {speaker}: unknown spoken_languages {unknown}")
            if outside := sorted(set(rules["required_languages"]) - set(rules["spoken_languages"])):
                errors.append(f"{label}: {speaker}: required_languages {outside} must be spoken_languages")
            if rules.get("within_turn_switching_required") and len(rules["required_languages"]) < 2:
                errors.append(f"{label}: {speaker}: within-turn switching needs two required languages")
        if not set(values) <= spoken:
            errors.append(f"{label}: value_surface_languages must be spoken by at least one speaker")
        scripts = profile["script_rules"]
        if "zh" in spoken and scripts["han_characters"] == "FORBIDDEN":
            errors.append(f"{label}: Han characters cannot be forbidden when Mandarin is spoken")
        if "zh" not in spoken and scripts["han_characters"] != "FORBIDDEN":
            errors.append(f"{label}: Han characters must be forbidden when no speaker speaks Mandarin")
        if scripts["latin_letters"] == "K01_IDENTIFIER_SPAN_ONLY" and "en" in spoken:
            errors.append(f"{label}: Latin letters can be limited to the K01 identifier only when no speaker speaks English")
        if scripts["arabic_digits"] == "FORBIDDEN" and "en" in values:
            errors.append(f"{label}: Arabic digits cannot be forbidden when values are rendered in English")
    for language in language_doc["terminology"]:
        if language != "note" and language not in languages:
            errors.append(f"{where}: terminology language {language} is not a release language")
    return errors


def _generation_plan_v2(documents):
    """Cross-file rules of a multilingual (2.0) generation plan: every count is exact within its language profile."""
    checklist, plan = documents["checklist"], documents["generation_plan"]
    catalog, semantics, normalization = documents["edge_case_catalog"], documents["check_semantics"], documents["normalization"]
    language_doc = documents["language_profiles"]
    errors = []
    where = "generation_plan"
    items = {item["id"]: item for item in checklist}
    compliance = [item["id"] for item in checklist if item["type"] == "COMPLIANCE"]
    fields = {item["field"]: item["id"] for item in checklist if item["type"] != "COMPLIANCE"}
    cases = {case["id"]: case for case in catalog["edge_cases"]}
    profile_of = {e["question_id"]: normalization["profiles"].get(e["normalization_profile"])
                  for e in semantics["checks"] if "normalization_profile" in e}
    profiles = language_doc["profiles"]
    plans = plan["language_profile_plans"]
    if [p["language_profile"] for p in plans] != list(profiles):
        errors.append(f"{where}: language_profile_plans must list every language profile of language_profiles.json once, in its order")
    if sum(p["num_calls"] for p in plans) != plan["num_calls"]:
        errors.append(f"{where}: language profile num_calls sum to {sum(p['num_calls'] for p in plans)}, not num_calls={plan['num_calls']}")
    durations = {d["profile"]: d for d in plan["duration_profiles"]}
    if len(durations) != len(plan["duration_profiles"]):
        errors.append(f"{where}: duplicate duration profile")
    for duration in plan["duration_profiles"]:
        if duration["min_seconds"] > duration["max_seconds"]:
            errors.append(f"{where}: {duration['profile']} min_seconds exceeds max_seconds")
    vocabulary = plan["call_intent_vocabulary"]
    rules = plan["planning_rules"]
    if unknown := sorted(set(rules["per_question_limit_exempt_edge_cases"]) - set(cases)):
        errors.append(f"{where}: per_question_limit_exempt_edge_cases references unknown edge cases {unknown}")
    prohibition = rules["prohibition_question_ids"]
    if any(items.get(q, {}).get("requirement_type") != "PROHIBITION" for q in prohibition):
        errors.append(f"{where}: prohibition_question_ids must name PROHIBITION compliance questions")
    if rules["load_cap"]["duration_profile"] not in durations:
        errors.append(f"{where}: load_cap names an unknown duration profile")
    dependents = []
    for rule in rules["dependent_fail_rules"]:
        q, failed, case_id = rule["question_id"], rule["fails_only_with"]["failed_question_id"], rule["fails_only_with"]["or_edge_case_id"]
        if q not in compliance or failed not in compliance or case_id not in cases:
            errors.append(f"{where}: dependent_fail_rules entry for {q} names unknown questions or edge cases")
        elif cases[case_id].get("verdict_requirements") != {failed: "PASS", q: "FAIL"}:
            errors.append(f"{where}: {case_id} must require {failed} PASS and {q} FAIL to license a dependent {q} FAIL")
        else:
            dependents.append((q, failed, case_id))

    extraction = plan["extraction_distribution"]
    vocab = {"transaction_direction": extraction["transaction_directions"], "order_type": extraction["order_types"],
             "transaction_currency": extraction["transaction_currencies"]}
    for field, values in vocab.items():
        profile = profile_of.get(fields.get(field))
        allowed = profile and profile["canonical"].get("allowed_values")
        if allowed is None or list(values) != allowed:
            errors.append(f"{where}: {field} vocabulary must equal normalization allowed_values {allowed}")
    currencies = extraction["transaction_currencies"]
    product_names = [product["name"] for product in extraction["product_pool"]]
    if len(product_names) != len(set(product_names)):
        errors.append(f"{where}: duplicate product name")
    for product in extraction["product_pool"]:
        if extra := set(product["allowed_currencies"]) - set(currencies):
            errors.append(f"{where}: {product['name']} allows unplanned currencies {sorted(extra)}")
    if list(extraction["transaction_amount_values_by_currency"]) != currencies:
        errors.append(f"{where}: transaction_amount_values_by_currency keys must equal transaction_currencies")
    if list(extraction["amount_quantity_coherence"]["ranges"]) != currencies:
        errors.append(f"{where}: amount_quantity_coherence ranges must cover exactly the transaction currencies")
    for currency in currencies:
        if not any(currency in p["allowed_currencies"] for p in extraction["product_pool"]):
            errors.append(f"{where}: no product allows currency {currency}")
        bounds = extraction["amount_quantity_coherence"]["ranges"].get(currency)
        if bounds and bounds["min"] > bounds["max"]:
            errors.append(f"{where}: {currency} unit-price range min exceeds max")
        elif not _coherent_pairs(extraction, currency):
            errors.append(f"{where}: {currency} has no coherent amount/quantity pair")
    names = (profile_of.get(fields.get("transaction_instrument")) or {}).get("language_names", {})
    if list(names) != product_names:
        errors.append(f"{where}: normalization language_names must cover exactly the product pool, in pool order")
    for p in plans:
        for language in profiles.get(p["language_profile"], {}).get("product_name_languages", []):
            if language == normalization["languages"][0]:
                continue
            for name in product_names:
                entry = names.get(name, {}).get(language)
                if entry is None or entry["status"] != "APPROVED":
                    errors.append(f"{where}: {p['language_profile']} needs an APPROVED {language} name for {name}")

    forms_by_case = {}
    for profile in normalization["profiles"].values():
        for form in profile["spoken_forms"]:
            if "enabled_by_edge_case" in form:
                forms_by_case.setdefault(form["enabled_by_edge_case"], []).append(form)
    for p in plans:
        name, n = p["language_profile"], p["num_calls"]
        label = f"{where}: {name}"
        language = profiles.get(name)
        if language is None:
            continue

        def require_total(what, counts):
            if sum(counts) != n:
                errors.append(f"{label}: {what} counts sum to {sum(counts)}, not num_calls={n}")

        if list(p["duration_counts"]) != list(durations):
            errors.append(f"{label}: duration_counts must list exactly the duration profiles")
        require_total("duration", p["duration_counts"].values())
        require_total("call intent set", [s["count"] for s in p["call_intent_set_counts"]])
        seen_sets = set()
        for entry in p["call_intent_set_counts"]:
            intents = entry["call_intents"]
            if any(i not in vocabulary for i in intents) or intents != sorted(intents, key=vocabulary.index):
                errors.append(f"{label}: call_intents {intents} must be vocabulary terms in vocabulary order")
            if tuple(intents) in seen_sets:
                errors.append(f"{label}: duplicate call intent set {intents}")
            seen_sets.add(tuple(intents))
        composition = p["call_composition_counts"]
        require_total("call composition", composition.values())
        all_pass = composition["all_pass_without_edge_cases"] + composition["all_pass_with_edge_cases"]
        some_fail = n - all_pass
        without_edges = composition["all_pass_without_edge_cases"] + composition["some_fail_without_edge_cases"]
        fail_hist = {int(k): v for k, v in p["fail_count_per_call_counts"].items()}
        require_total("fail_count_per_call", fail_hist.values())
        if fail_hist.get(0, 0) != all_pass:
            errors.append(f"{label}: calls with 0 FAILs ({fail_hist.get(0, 0)}) must equal all-PASS calls ({all_pass})")
        if max(fail_hist) > len(compliance):
            errors.append(f"{label}: a call cannot have more FAILs than the {len(compliance)} compliance questions")
        fails = p["fail_counts_per_check"]
        if list(fails) != compliance:
            errors.append(f"{label}: fail_counts_per_check must list exactly the compliance questions in checklist order")
            continue
        if sum(fails.values()) != sum(k * v for k, v in fail_hist.items()):
            errors.append(f"{label}: fail_counts_per_check total {sum(fails.values())} must equal the per-call FAIL total "
                          f"{sum(k * v for k, v in fail_hist.items())}")
        if over := sorted(q for q, count in fails.items() if count > some_fail):
            errors.append(f"{label}: {over} FAIL in more calls than the {some_fail} some-FAIL calls")
        edge_hist = {int(k): v for k, v in p["edge_cases_per_call_counts"].items()}
        require_total("edge_cases_per_call", edge_hist.values())
        if edge_hist.get(0, 0) != without_edges:
            errors.append(f"{label}: calls with 0 edge cases ({edge_hist.get(0, 0)}) must equal calls without edge cases ({without_edges})")
        counts = p["edge_case_counts"]
        if list(counts) != list(cases):
            errors.append(f"{label}: edge_case_counts must list every catalog edge case in catalog order")
            continue
        if sum(counts.values()) != sum(k * v for k, v in edge_hist.items()):
            errors.append(f"{label}: edge_case_counts total {sum(counts.values())} must equal the per-call edge-case total "
                          f"{sum(k * v for k, v in edge_hist.items())}")
        for case_id, count in counts.items():
            if count and name not in cases[case_id].get("language_profiles", []):
                errors.append(f"{label}: {case_id} is not eligible for this language profile")
            for form in forms_by_case.get(case_id, []) if count else []:
                if form.get("language") not in language["value_surface_languages"]:
                    errors.append(f"{label}: {case_id} requires {form['language']} form {form['id']}, which this profile cannot speak")
        if composition["all_pass_with_edge_cases"] > sum(
                count for case_id, count in counts.items() if "FAIL" not in cases[case_id].get("verdict_requirements", {}).values()):
            errors.append(f"{label}: too few edge cases that force no FAIL for the all-PASS calls with edge cases")
        for case_id, by_variant in p["language_variant_counts"].items():
            case = cases.get(case_id)
            defined = {v["variant_id"]: v for v in (case or {}).get("language_variants", [])}
            for variant_id in by_variant:
                variant = defined.get(variant_id)
                if variant is None or variant["status"] != "READY_V1" or name not in variant["profiles"]:
                    errors.append(f"{label}: {case_id}/{variant_id} is not a READY_V1 variant for this profile")
            if case is not None and sum(by_variant.values()) > counts[case_id]:
                errors.append(f"{label}: {case_id} has more variants than assignments")
        styles = language.get("mixing_styles")
        if ("mixing_style_counts" in p) != (styles is not None):
            errors.append(f"{label}: mixing_style_counts are required exactly for profiles that define mixing_styles")
        elif styles is not None:
            if list(p["mixing_style_counts"]) != list(styles):
                errors.append(f"{label}: mixing_style_counts must list exactly the profile's mixing styles")
            require_total("mixing style", p["mixing_style_counts"].values())
        for key, field in (("transaction_direction_counts", "transaction_direction"), ("order_type_counts", "order_type"),
                           ("transaction_currency_counts", "transaction_currency")):
            if list(p[key]) != list(vocab[field]):
                errors.append(f"{label}: {key} must list exactly the {field} vocabulary in order")
            require_total(key, p[key].values())
        for q, cap in p["target_exposure_caps"].items():
            if q not in items or cap > n:
                errors.append(f"{label}: target_exposure_caps {q}={cap} must name a question and not exceed num_calls")
        forced_fail, forced_pass = {}, {}
        for case_id, count in counts.items():
            for q, verdict in cases[case_id].get("verdict_requirements", {}).items():
                target = forced_fail if verdict == "FAIL" else forced_pass
                target[q] = target.get(q, 0) + count
        for q in compliance:
            if forced_fail.get(q, 0) > fails[q]:
                errors.append(f"{label}: {forced_fail[q]} edge cases force {q} FAIL but only {fails[q]} FAILs are planned")
            if forced_pass.get(q, 0) > n - fails[q]:
                errors.append(f"{label}: {forced_pass[q]} edge cases force {q} PASS but only {n - fails[q]} PASSes are planned")
        for q, failed, case_id in dependents:
            if fails[q] != fails[failed] + counts[case_id]:
                errors.append(f"{label}: {q} FAILs ({fails[q]}) must equal {failed} FAILs ({fails[failed]}) plus {case_id} "
                              f"assignments ({counts[case_id]})")
        if sum(fails[q] for q in prohibition if q in fails) > rules["max_prohibition_fails_per_call"] * some_fail:
            errors.append(f"{label}: planned prohibition FAILs exceed {rules['max_prohibition_fails_per_call']} per some-FAIL call")
        planned_currencies = [c for c, count in p["transaction_currency_counts"].items() if count]
        for case_id, count in counts.items():
            prerequisite = cases[case_id].get("gold_prerequisites")
            if not count or prerequisite is None or not prerequisite["kind"].startswith("NUMERIC_"):
                continue
            for target in cases[case_id]["allowed_target_question_ids"]:
                index = {"transaction_amount": 0, "transaction_quantity": 1}.get(next((f for f, q in fields.items() if q == target), None))
                for currency in planned_currencies if index is not None else []:
                    if not any(numeric_prerequisite_met(prerequisite, pair[index]) for pair in _coherent_pairs(extraction, currency)):
                        errors.append(f"{label}: no coherent {currency} value satisfies {case_id} prerequisites")
        for case_id, by_variant in p["language_variant_counts"].items():
            for variant in cases.get(case_id, {}).get("language_variants", []):
                constraint = variant.get("constraint")
                if variant["variant_id"] not in by_variant or constraint is None:
                    continue
                factor = constraint["factor"]
                for currency in planned_currencies:
                    amounts = extraction["transaction_amount_values_by_currency"][currency]
                    if not any(a * factor in amounts or (a % factor == 0 and a // factor in amounts)
                               for a, _ in _coherent_pairs(extraction, currency)):
                        errors.append(f"{label}: no coherent {currency} amount has a x{factor} partner for {case_id}/{variant['variant_id']}")

    kyc = extraction["synthetic_kyc"]
    identifier = profile_of.get(fields.get("client_identification_number"))
    pattern = identifier and identifier["canonical"].get("pattern")
    nric = kyc["nric_fin"]
    if pattern:
        for prefix in nric["prefixes"]:
            for suffix in nric["suffix_letters"]:
                if not re.fullmatch(pattern, prefix + "0" * nric["digit_count"] + suffix):
                    errors.append(f"{where}: NRIC/FIN prefix {prefix} with suffix {suffix} does not fit the canonical pattern")
                    break
    excluded = (identifier or {}).get("canonical", {}).get("excluded_characters", [])
    if clash := sorted(set(excluded) & (set(nric["suffix_letters"]) | set(nric["prefixes"]))):
        errors.append(f"{where}: NRIC/FIN pools contain excluded characters {clash}")
    if len(set(nric["suffix_letters"])) != len(nric["suffix_letters"]):
        errors.append(f"{where}: suffix_letters must not repeat")
    try:
        first, last = date.fromisoformat(kyc["dob"]["min_date"]), date.fromisoformat(kyc["dob"]["max_date"])
    except ValueError as error:
        errors.append(f"{where}: invalid DOB range date: {error}")
    else:
        if (last - first).days + 1 < plan["num_calls"]:
            errors.append(f"{where}: DOB range cannot provide {plan['num_calls']} unique dates")
    return errors


def validate_stage0(documents, schema_dir=DEFAULT_SCHEMA_DIR):
    """Return errors for already-loaded Stage 0 documents; never mutate them."""
    errors = _structural(documents, schema_dir)
    if errors:
        return errors
    if errors := _versions(documents):
        return errors
    errors = _no_applicability(documents)
    checklist, semantics = documents["checklist"], documents["check_semantics"]
    normalization, catalog = documents["normalization"], documents["edge_case_catalog"]
    errors += _check_semantics(checklist, semantics, normalization)
    errors += _normalization(normalization, catalog, semantics)
    errors += _catalog(checklist, catalog, semantics, normalization)
    if is_multilingual(documents):
        errors += _catalog_languages(catalog, documents["language_profiles"], normalization)
        errors += _language_profiles(documents["language_profiles"], normalization)
        errors += _generation_plan_v2(documents)
    else:
        errors += _monolingual_catalog(catalog)
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
