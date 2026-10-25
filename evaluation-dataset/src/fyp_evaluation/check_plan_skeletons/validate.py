"""Independent G2B validation of a skeleton set against the current call specs and G0.

Works on {relative path: text} so the same checks run on freshly compiled in-memory texts and
on the files actually on disk. Two layers:

  1. invariants re-derived here from the call spec and the frozen G0 documents (evidence mode,
     condition references, FINAL events, value and spoken-form rules, constraint consistency,
     the G0 ordering rules); the compiler's own code is not reused for them; and
  2. exact canonical bytes: each file must equal the serialization rebuilt from the inputs, so a
     reordered key or a hand edit fails even when it is semantically equivalent, and the record is
     never an authority.
"""

from pathlib import Path

from jsonschema import Draft202012Validator

from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.planning import outputs as g1
from fyp_evaluation.validation.schemas import load_json
from fyp_evaluation.validation.stage0 import DEFAULT_SCHEMA_DIR

from .compile import RECORD, build_record, build_skeletons, skeleton_path
from .model import KIND_STATUSES, VALUE_KINDS, same_scalar

SCHEMA_NAMES = ("check_plan_skeleton", "check_plan_skeleton_record")
BASE_FIELDS = {"event_id", "kind", "status", "allowed_speakers", "question_ids", "assignment_ids"}
REQUIRED_FIELDS = {
    "COMPLIANCE_ACT": {"role", "condition_ref"}, "VALUE_STATEMENT": {"values"},
    "VALUE_CORRECTION": {"values", "corrects_event_id"}, "READBACK": {"readback"}, "ORDER_STATEMENT": {"values"},
    "ORDER_INVALIDATION": {"invalidates_event_id"}, "DISTRACTOR_MENTION": {"values", "distractor_role"},
    "ANTECEDENT_MENTION": {"values"}, "REFERENCE_SELECTION": {"values", "refers_to_event_id"},
    "FALSE_START": {"incomplete"}, "GUARANTEE_REQUEST": set(), "GUARANTEE_REFUSAL": set(),
    "WEAK_ACKNOWLEDGEMENT": {"strength"},
}
OPTIONAL_FIELDS = {"DISTRACTOR_MENTION": {"currency"}}


def _validator(schema_dir, name):
    schema = load_json(Path(schema_dir) / "stage2" / f"{name}.schema.json")
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def check_skeleton_schemas(schema_dir=DEFAULT_SCHEMA_DIR):
    for name in SCHEMA_NAMES:
        _validator(schema_dir, name)


def _same(left, right):
    return g1.canonical_json(left) == g1.canonical_json(right)


def _cycle(edges, nodes):
    """True when the directed ORDER graph contains a cycle."""
    successors = {node: [] for node in nodes}
    for before, after in edges:
        successors[before].append(after)
    state = {}

    def visit(node):
        state[node] = 1
        for nxt in successors[node]:
            if state.get(nxt) == 1 or (nxt not in state and visit(nxt)):
                return True
        state[node] = 2
        return False
    return any(node not in state and visit(node) for node in nodes)


CROSS_LANGUAGE_ROLES = {
    # case: (earlier-state events, later-state events), re-derived here independently of languages.py
    "EC_CORRECTION_SAME_TURN": ("SUPERSEDED", "FINAL"), "EC_CORRECTION_CROSS_TURN": ("SUPERSEDED", "FINAL"),
    "EC_MULTIPLE_CORRECTIONS": ("FIRST_SUPERSEDED", "FINAL"),
    "EC_EMPLOYEE_READBACK_ERROR": ("VALUE_STATEMENT", "READBACK"), "EC_ONE_FIELD_READBACK_ERROR": ("VALUE_STATEMENT", "READBACK"),
    "EC_COMPLETE_ORDER_REPLACEMENT": ("SUPERSEDED_ORDER", "FINAL_ORDER"),
    "EC_PREVIOUS_CURRENCY_REFERENCE": ("ANTECEDENT_MENTION", "REFERENCE_SELECTION"),
    "EC_PRODUCT_REFERENCE": ("ANTECEDENT_MENTION", "REFERENCE_SELECTION"),
    "EC_PREVIOUS_QUANTITY_REFERENCE": ("ANTECEDENT_MENTION", "REFERENCE_SELECTION"),
    "EC_MULTIPLE_CURRENCIES": ("DISTRACTOR_MENTION", "TARGET_FINAL"), "EC_MULTIPLE_MONETARY_VALUES": ("DISTRACTOR_MENTION", "TARGET_FINAL"),
    "EC_FALSE_START": ("FALSE_START", "TARGET_FINAL"),
}


def _expected_pins(events, assignments, catalog, checks):
    """{event_id: language} required by the assigned language variants, and any contradiction found."""
    pins, errors = {}, []
    for a in assignments.values():
        variant = next((v for v in catalog[a["edge_case_id"]].get("language_variants", [])
                        if v["variant_id"] == a.get("language_variant")), None)
        languages = (variant or {}).get("event_languages", {})
        if not languages:
            continue
        own = [e for e in events if a["assignment_id"] in e["assignment_ids"]]
        if a["edge_case_id"] in CROSS_LANGUAGE_ROLES:
            early, late = CROSS_LANGUAGE_ROLES[a["edge_case_id"]]

            def pick(role):
                if role == "FIRST_SUPERSEDED":
                    return [e["event_id"] for e in own if e["status"] == "SUPERSEDED"][:1]
                if role == "SUPERSEDED_ORDER":
                    return [e["event_id"] for e in own if e["kind"] == "ORDER_STATEMENT" and e["status"] == "SUPERSEDED"]
                if role == "FINAL_ORDER":
                    return [e["event_id"] for e in own if e["kind"] == "ORDER_STATEMENT" and e["status"] == "FINAL"]
                if role == "TARGET_FINAL":
                    return [checks[q]["final_event_id"] for q in a["target_question_ids"]]
                if role in {"SUPERSEDED", "FINAL"}:
                    return [e["event_id"] for e in own if e["status"] == role]
                return [e["event_id"] for e in own if e["kind"] == role]
            wanted = [(event_id, languages["source"]) for event_id in pick(early)] + [(event_id, languages["target"]) for event_id in pick(late)]
        else:
            wanted = [(e["event_id"], languages["target"]) for e in events for q in a["target_question_ids"]
                      if e.get("values", {}).get(q, {}).get("spoken_form", {}).get("requirement") == "REQUIRED"]
        if not wanted:
            errors.append(f"{a['assignment_id']}: variant {a['language_variant']} pins no events")
        for event_id, language in wanted:
            if pins.setdefault(event_id, language) != language:
                errors.append(f"{event_id}: language variants require two different languages")
    return pins, errors


def _skeleton_errors(sk, call_id, inputs, validator):
    errors = [f"{e.json_path}: {e.message}" for e in validator.iter_errors(sk)]
    if errors:
        return errors
    add = errors.append
    docs, spec = inputs.documents, inputs.specs[call_id]
    checklist = docs["checklist"]
    items = {i["id"]: i for i in checklist}
    sem = {e["question_id"]: e for e in docs["check_semantics"]["checks"]}
    catalog = {c["id"]: c for c in docs["edge_case_catalog"]["edge_cases"]}
    profiles = docs["normalization"]["profiles"]
    expected = {g["question_id"]: g["expected"] for g in spec["gold"]}
    gold = {q: e.get("expected_verdict", e.get("expected_value")) for q, e in expected.items()}
    assignments = {a["assignment_id"]: a for a in spec["edge_cases"]}
    requirements = spec.get("language_requirements")
    multilingual = requirements is not None

    def spoken(profile, form):
        """A form this call may speak (monolingual releases: every form)."""
        if not multilingual:
            return True
        key = "product_name_languages" if "language_names" in profile else "value_surface_languages"
        return form.get("language") in requirements[key]

    if sk["call_id"] != call_id:
        add(f"call_id {sk['call_id']} does not match the file name")
    if sk["dataset_id"] != inputs.dataset_id:
        add("dataset_id differs from dataset_manifest.json")
    if sk["call_spec_sha256"] != inputs.spec_hashes[call_id]:
        add("call_spec_sha256 differs from the current call spec")
    if not _same(sk["source"], inputs.g0_hashes):
        add("source hashes differ from the current G0 files")
    if sk["schema_version"] != spec["schema_version"]:
        add(f"schema_version {sk['schema_version']} differs from the call spec's {spec['schema_version']}")

    events, checks_list = sk["planned_events"], sk["checks"]
    if [c["question_id"] for c in checks_list] != [i["id"] for i in checklist]:
        return errors + [f"checks must list exactly the {len(checklist)} checklist questions in checklist order"]
    checks = {c["question_id"]: c for c in checks_list}
    if [e["event_id"] for e in events] != [f"EV_{n:03d}" for n in range(1, len(events) + 1)]:
        add("event IDs must be EV_001.. in order")
    by_id = {e["event_id"]: e for e in events}

    # ---- events: vocabulary, per-kind fields, references --------------------------------------------
    for e in events:
        label = e["event_id"]
        if e["status"] not in KIND_STATUSES[e["kind"]]:
            add(f"{label}: {e['kind']} cannot have status {e['status']}")
        present = set(e) - BASE_FIELDS
        required, optional = REQUIRED_FIELDS[e["kind"]], OPTIONAL_FIELDS.get(e["kind"], set())
        if multilingual:
            optional = optional | {"required_language"}
        if present - required - optional or required - present:
            add(f"{label}: {e['kind']} fields {sorted(present)} must be {sorted(required)} (+ optional {sorted(optional)})")
            continue
        if any(q not in items for q in e["question_ids"]):
            add(f"{label}: unknown question in {e['question_ids']}")
        if any(a not in assignments for a in e["assignment_ids"]):
            add(f"{label}: references an assignment missing from the call spec")
        for field in ("corrects_event_id", "invalidates_event_id", "refers_to_event_id"):
            if field in e:
                target = by_id.get(e[field])
                if target is None or int(target["event_id"][3:]) >= int(label[3:]):
                    add(f"{label}: {field} must name an earlier existing event")
        if e["kind"] == "ORDER_INVALIDATION" and by_id.get(e["invalidates_event_id"], {}).get("kind") != "ORDER_STATEMENT":
            add(f"{label}: must invalidate an ORDER_STATEMENT")
        if e["kind"] == "REFERENCE_SELECTION" and by_id.get(e["refers_to_event_id"], {}).get("kind") != "ANTECEDENT_MENTION":
            add(f"{label}: must refer to an ANTECEDENT_MENTION")
        if e["kind"] in VALUE_KINDS and set(e["values"]) != set(e["question_ids"]):
            add(f"{label}: values must cover exactly its question_ids")

    # ---- compliance checks: re-derive mode, pattern and references ---------------------------------
    for item in checklist:
        q, c = item["id"], checks[item["id"]]
        if item["type"] != "COMPLIANCE":
            continue
        if c["kind"] != "COMPLIANCE" or not _same(c["expected"], expected[q]):
            add(f"{q}: expected answer differs from the call spec")
            continue
        presence = (item["requirement_type"] == "OBLIGATION") == (gold[q] == "PASS")
        mode = "PRESENCE" if presence else "ABSENCE"
        ref = f"{q}.{'pass_condition' if item['requirement_type'] == 'OBLIGATION' else 'fail_condition'}"
        pattern = sem[q]["presence_pattern"]
        if c["evidence_mode"] != mode:
            add(f"{q}: evidence_mode {c['evidence_mode']} must be {mode} for {item['requirement_type']} {gold[q]}")
        if c["presence_pattern"] != pattern:
            add(f"{q}: presence_pattern {c['presence_pattern']} must be {pattern}")
        if presence:
            if c.get("required_condition_ref") != ref:
                add(f"{q}: required_condition_ref must be {ref}")
            acts = [e for e in events if e["kind"] == "COMPLIANCE_ACT" and e["question_ids"] == [q]
                    and e["condition_ref"] == ref and e["event_id"] in c["evidence_event_ids"]]
            speakers = {"EMPLOYEE_ONLY": [(["EMPLOYEE"], "ACT")], "ANY_SPEAKER": [(["CLIENT", "EMPLOYEE"], "ACT")],
                        "EXCHANGE": [(["EMPLOYEE"], "ACT"), (["CLIENT"], "RESPONSE")]}[pattern]
            if [(e["allowed_speakers"], e["role"]) for e in acts] != speakers or c["evidence_event_ids"] != [e["event_id"] for e in acts]:
                add(f"{q}: {pattern} presence evidence must be exactly {speakers}")
        else:
            if c.get("forbidden_condition_ref") != ref:
                add(f"{q}: forbidden_condition_ref must be {ref}")
            if c.get("absence") != {"scope": "ENTIRE_DIALOGUE"}:
                add(f"{q}: absence scope must be ENTIRE_DIALOGUE")
            if c["evidence_event_ids"]:
                add(f"{q}: an ABSENCE check lists no evidence events")
        wording = [a for a in assignments.values() if q in a["target_question_ids"]
                   and "wording_constraint" in catalog[a["edge_case_id"]]["plan_template"]]
        if ("wording_constraint" in c) != bool(wording) or (wording and c["wording_constraint"] != catalog[wording[0]["edge_case_id"]]["plan_template"]["wording_constraint"]):
            add(f"{q}: wording_constraint must appear exactly when a paraphrase edge case targets it")
    # ---- extraction checks: FINAL event, values, spoken forms --------------------------------------
    edge_forms = {}
    for a in assignments.values():
        for q in a["target_question_ids"]:
            profile = profiles.get(sem.get(q, {}).get("normalization_profile"))
            for form in (profile or {}).get("spoken_forms", []):
                if form.get("enabled_by_edge_case") == a["edge_case_id"]:
                    edge_forms[q] = (form, a)
    for item in checklist:
        q = item["id"]
        if item["type"] == "COMPLIANCE":
            continue
        c, profile = checks[q], profiles[sem[q]["normalization_profile"]]
        if c["kind"] != "EXTRACTION" or not _same(c["expected"], expected[q]):
            add(f"{q}: expected answer differs from the call spec")
            continue
        if c["resolution_ref"] != f"{q}.resolution_rule":
            add(f"{q}: resolution_ref must be {q}.resolution_rule")
        finals = [e["event_id"] for e in events if e["status"] == "FINAL" and q in e["question_ids"]]
        if len(finals) != 1:
            add(f"{q}: exactly one FINAL event is required, found {finals}")
        else:
            if c["final_event_id"] != finals[0]:
                add(f"{q}: final_event_id must be {finals[0]}")
            entry = by_id[finals[0]]["values"][q]
            if not same_scalar(entry["value"], gold[q]):
                add(f"{q}: FINAL value {entry['value']!r} differs from gold {gold[q]!r}")
        if c["evidence_event_ids"] != [e["event_id"] for e in events if q in e["question_ids"]]:
            add(f"{q}: evidence_event_ids must list every event concerning the question, in order")
        permitted = {"requirement": "ONE_OF", "forms": [f["id"] for f in profile["spoken_forms"]
                                                        if f["usage"] in {"DEFAULT", "PERMITTED"} and spoken(profile, f)]}
        if not _same(c["non_gold_spoken_form"], permitted):
            add(f"{q}: non_gold_spoken_form must be the default/permitted forms")
        assigned = {edge_forms[q][0]["id"]} if q in edge_forms else set()
        forbidden = [f["id"] for f in profile["spoken_forms"] if f["usage"] == "EDGE_CASE_ONLY" and f["id"] not in assigned]
        if c["forbidden_spoken_forms"] != forbidden:
            add(f"{q}: forbidden_spoken_forms must be the unassigned edge-case-only forms {forbidden}")
        restatement = c["gold_restatement"]
        if q in edge_forms:
            form, a = edge_forms[q]
            required_form = restatement["spoken_form"]
            ok = (restatement["policy"] == "ASSIGNED_FORM_ONLY" and required_form.get("requirement") == "REQUIRED"
                  and required_form.get("form") == form["id"] and required_form.get("applies_to") == form["applies_to"]
                  and required_form.get("variant") == a["parameters"].get("variant")
                  and required_form.get("text") == a["parameters"].get("spoken_form"))
            if not ok:
                add(f"{q}: gold_restatement must require the assigned form {form['id']} with its locked parameters")
        elif restatement["policy"] != "ANY_PERMITTED_FORM" or not _same(restatement["spoken_form"], permitted):
            add(f"{q}: gold_restatement must allow the default/permitted forms when no spoken-form edge case is assigned")
        # every value mention
        for e in events:
            entry = e.get("values", {}).get(q)
            if entry is None:
                continue
            equals_gold = same_scalar(entry["value"], gold[q])
            if e["kind"] == "REFERENCE_SELECTION":
                if not equals_gold or entry["spoken_form"] != {"requirement": "NOT_EXPLICIT"}:
                    add(f"{e['event_id']}: a reference selection resolves to gold without restating it")
            elif equals_gold:
                if e["status"] in {"SUPERSEDED", "DISTRACTOR"} and e["kind"] != "ORDER_STATEMENT":
                    add(f"{e['event_id']}: a {e['status']} value equals gold for {q}")
                if not _same(entry["spoken_form"], restatement["spoken_form"]):
                    add(f"{e['event_id']}: a gold-valued {q} mention must use the gold_restatement form")
            else:
                if e["status"] in {"FINAL", "STATED"}:
                    add(f"{e['event_id']}: a {e['status']} value for {q} must equal gold")
                if not _same(entry["spoken_form"], permitted):
                    add(f"{e['event_id']}: a non-gold {q} value may use only default/permitted forms")
            if entry["spoken_form"].get("requirement") == "REQUIRED" and q not in edge_forms:
                add(f"{e['event_id']}: required spoken form for {q} without an assigned spoken-form edge case")
        for e in events:
            readback = e.get("readback")
            if readback and readback["wrong_question_id"] == q:
                if same_scalar(readback["wrong_value"], gold[q]) or not _same(readback["wrong_value_spoken_form"], permitted):
                    add(f"{e['event_id']}: readback wrong value must differ from gold and use default/permitted forms")
    for e in events:
        readback = e.get("readback")
        if readback and (readback["wrong_question_id"] not in readback["must_include_question_ids"]
                         or readback["wrong_question_id"] not in e["question_ids"]):
            add(f"{e['event_id']}: the wrong field must be among the fields the readback must include")

    # ---- language variants: exactly the pins their catalog definitions require ------------------------
    if multilingual:
        pins, pin_errors = _expected_pins(events, assignments, catalog, checks)
        errors += pin_errors
        actual = {e["event_id"]: e["required_language"] for e in events if "required_language" in e}
        if actual != pins:
            add(f"required_language pins {actual} differ from the language variants' pins {pins}")

    # ---- edge assignments: represented, parameters respected ---------------------------------------
    mentioned = {a for e in events for a in e["assignment_ids"]} | {a for c in checks_list for a in c["assignment_ids"]}
    if missing := sorted(set(assignments) - mentioned):
        add(f"edge-case assignments not represented in the skeleton: {missing}")
    if stray := sorted(mentioned - set(assignments)):
        add(f"skeleton references assignments absent from the call spec: {stray}")
    constraints = sk["constraints"]
    keys = [g1.canonical_json(c) for c in constraints]
    if len(keys) != len(set(keys)):
        add("duplicate constraints")
    ids = set(by_id)
    for c in constraints:
        referenced = [c["before"], c["after"]] if c["type"] == "ORDER" else c["events"]
        if any(r not in ids for r in referenced) or len(set(referenced)) != 2:
            add(f"constraint {c['type']} {referenced} must reference two distinct existing events")
        source = c["source"]
        if source["type"] == "EDGE_ASSIGNMENT" and source["id"] not in assignments:
            add(f"constraint source {source['id']} is not an assignment of this call")
        if source["type"] == "ORDERING_RULE" and source["id"] not in {r["id"] for r in docs["check_semantics"]["ordering_rules"]}:
            add(f"constraint source {source['id']} is not a G0 ordering rule")
        if source["type"] == "PRESENCE_PATTERN" and sem.get(source["question_id"], {}).get("presence_pattern") != "EXCHANGE":
            add(f"PRESENCE_PATTERN constraint on {source['question_id']}, which is not an EXCHANGE check")
    if not any(error.startswith("constraint ") for error in errors):
        if _cycle([(c["before"], c["after"]) for c in constraints if c["type"] == "ORDER"], sorted(ids)):
            add("ORDER constraints contain a cycle")
        group = {i: i for i in ids}

        def find(node):
            while group[node] != node:
                group[node] = group[group[node]]
                node = group[node]
            return node
        for c in constraints:
            if c["type"] == "SAME_TURN":
                group[find(c["events"][0])] = find(c["events"][1])
        for c in constraints:
            if c["type"] in {"NOT_SAME_TURN", "MIN_INTERVENING_TURNS"} and find(c["events"][0]) == find(c["events"][1]):
                add(f"{c['type']} {c['events']} contradicts SAME_TURN constraints")

    def has(kind, events_, source_id, **extra):
        return any(c["type"] == kind and c["source"] == {"type": "EDGE_ASSIGNMENT", "id": source_id}
                   and (c.get("events") == sorted(events_) if events_ else True) and all(c.get(k) == v for k, v in extra.items())
                   for c in constraints)
    for a in assignments.values():
        template = catalog[a["edge_case_id"]]["plan_template"]
        if "same_turn" in template and not any(
                c["type"] == ("SAME_TURN" if template["same_turn"] else "NOT_SAME_TURN") and c["source"]["id"] == a["assignment_id"]
                for c in constraints if c["source"]["type"] == "EDGE_ASSIGNMENT"):
            add(f"{a['assignment_id']}: missing {'SAME_TURN' if template['same_turn'] else 'NOT_SAME_TURN'} from its catalog template")
        if template.get("min_intervening_turns") and not has("MIN_INTERVENING_TURNS", None, a["assignment_id"], turns=template["min_intervening_turns"]):
            add(f"{a['assignment_id']}: missing MIN_INTERVENING_TURNS {template['min_intervening_turns']}")
        if a["edge_case_id"] == "EC_LONG_DISTANCE_DEPENDENCY":
            finals = sorted(by_id[checks[q]["final_event_id"]]["event_id"] for q in a["target_question_ids"])
            if len(set(finals)) < len(finals):
                add(f"{a['assignment_id']}: unsatisfiable long-distance dependency (shared FINAL event {finals})")
            elif not has("MIN_INTERVENING_TURNS", finals, a["assignment_id"], turns=a["parameters"]["min_turn_gap"]):
                add(f"{a['assignment_id']}: missing MIN_INTERVENING_TURNS {a['parameters']['min_turn_gap']} between {finals}")
        if a["edge_case_id"] == "EC_PRODUCT_REFERENCE":
            ante = [e for e in events if e["kind"] == "ANTECEDENT_MENTION" and a["assignment_id"] in e["assignment_ids"]]
            if [e["values"][a["target_question_ids"][0]]["value"] for e in ante] != a["parameters"]["candidate_products"]:
                add(f"{a['assignment_id']}: candidate_products must be mentioned in the locked order")
        if "antecedent_speaker" in a["parameters"]:
            for e in events:
                if e["kind"] == "ANTECEDENT_MENTION" and a["assignment_id"] in e["assignment_ids"] and e["allowed_speakers"] != [a["parameters"]["antecedent_speaker"]]:
                    add(f"{e['event_id']}: antecedent speaker must be the G1-locked {a['parameters']['antecedent_speaker']}")
        if "allowed_distractor_speakers" in catalog[a["edge_case_id"]]:
            for e in events:
                if e["kind"] == "DISTRACTOR_MENTION" and a["assignment_id"] in e["assignment_ids"] and e["allowed_speakers"] != sorted(catalog[a["edge_case_id"]]["allowed_distractor_speakers"]):
                    add(f"{e['event_id']}: distractor speakers must be the G0 allowed speakers")

    # ---- G0 ordering rules: re-derived, and nothing else is an ORDERING_RULE ------------------------
    wanted = set()
    for rule in docs["check_semantics"]["ordering_rules"]:
        subjects = [e["event_id"] for e in events if e["kind"] == "COMPLIANCE_ACT" and e["role"] == "ACT" and rule["question_id"] in e["question_ids"]]
        after = rule["after"]
        if after["type"] == "CHECK_EVIDENCE":
            refs = [e["event_id"] for e in events if e["kind"] == "COMPLIANCE_ACT" and after["question_id"] in e["question_ids"]]
        else:
            refs = {checks[q]["final_event_id"] for q in after["question_ids"]}
        wanted |= {(rule["id"], r, s) for r in refs for s in subjects}
    actual = {(c["source"]["id"], c["before"], c["after"]) for c in constraints if c["type"] == "ORDER" and c["source"]["type"] == "ORDERING_RULE"}
    if wanted != actual:
        add(f"G0 ordering-rule constraints differ: missing {sorted(wanted - actual)}, unexpected {sorted(actual - wanted)}")
    for item in checklist:
        if item["type"] == "COMPLIANCE" and sem[item["id"]]["presence_pattern"] == "EXCHANGE" and checks[item["id"]]["evidence_mode"] == "PRESENCE":
            act, response = checks[item["id"]]["evidence_event_ids"]
            if not any(c["type"] == "ORDER" and (c["before"], c["after"]) == (act, response)
                       and c["source"] == {"type": "PRESENCE_PATTERN", "question_id": item["id"]} for c in constraints):
                add(f"{item['id']}: EXCHANGE presence needs ORDER({act}, {response})")
    return errors


def validate_skeleton_set(inputs, texts, schema_dir=DEFAULT_SCHEMA_DIR):
    """Return every violation found in a set of skeleton texts ([] when valid)."""
    skeleton_validator = _validator(schema_dir, "check_plan_skeleton")
    record_validator = _validator(schema_dir, "check_plan_skeleton_record")
    expected_paths = {skeleton_path(call_id): call_id for call_id in inputs.call_ids}
    try:
        canonical = build_skeletons(inputs)
    except ValueError as error:
        return [f"cannot rebuild the canonical skeletons from the call specs: {error}"]
    canonical_texts = {sk["call_id"]: serialize_json(sk) for sk in canonical}
    errors = []
    for rel in sorted(texts):
        if rel == RECORD or rel in expected_paths:
            continue
        hint = "; orphaned temporary file from an interrupted run, rerun with --overwrite" if rel.endswith(".tmp") else ""
        errors.append(f"unexpected file {rel}{hint}")
    errors += [f"missing {rel}" for rel in expected_paths if rel not in texts]
    for rel, call_id in expected_paths.items():
        if rel not in texts:
            continue
        if texts[rel] is None:
            errors.append(f"{rel}: unreadable or not a regular file")
            continue
        try:
            skeleton = parse_json_text(texts[rel])
        except ValueError as error:
            errors.append(f"{rel}: invalid JSON: {error}")
            continue
        problems = _skeleton_errors(skeleton, call_id, inputs, skeleton_validator)
        if problems:
            errors += [f"{rel}: {problem}" for problem in problems]
        elif texts[rel] != canonical_texts[call_id]:
            errors.append(f"{rel}: not the canonical serialization rebuilt from the call spec and G0 (key order or formatting)")

    if RECORD not in texts:
        errors.append(f"missing {RECORD}")
    elif texts[RECORD] is None:
        errors.append(f"{RECORD}: unreadable or not a regular file")
    else:
        try:
            record = parse_json_text(texts[RECORD])
        except ValueError as error:
            errors.append(f"{RECORD}: invalid JSON: {error}")
        else:
            shape = [f"{e.json_path}: {e.message}" for e in record_validator.iter_errors(record)]
            errors += [f"{RECORD}: {problem}" for problem in shape]
            if not shape:
                rebuilt = build_record(inputs, canonical, canonical_texts)
                for key in rebuilt:
                    if not _same(record[key], rebuilt[key]):
                        errors.append(f"{RECORD}: {key} differs from the value recomputed from the call specs and G0")
                if texts[RECORD] != serialize_json(rebuilt):
                    errors.append(f"{RECORD}: not the canonical serialization rebuilt from the call specs and G0")
    return errors
