"""Python-owned product references: deterministic first-seen order, deduplicated by name, roles unioned.

Order of discovery (never a set, a set-built dict or alphabetical order):
  1. the final T01 gold product;
  2. edge assignments in their stored call-spec order;
  3. within an assignment, parameter values in stored parameter order (list values in stored order);
  4. deduplication by exact product name with the union of every role the product plays;
  5. P1, P2, ... in that resulting order.

A T01-valued parameter that is not explicitly mapped here fails loudly instead of being guessed.
"""

from .config import PRODUCT_ROLES

# edge_case_id -> {parameter name: role}. Used when the assignment targets T01.
_TARGETED_PARAMETERS = {
    "EC_CORRECTION_SAME_TURN": {"initial_value": "SUPERSEDED", "final_value": "FINAL"},
    "EC_CORRECTION_CROSS_TURN": {"initial_value": "SUPERSEDED", "final_value": "FINAL"},
    "EC_MULTIPLE_CORRECTIONS": {"superseded_values": "SUPERSEDED", "final_value": "FINAL"},
    "EC_EMPLOYEE_READBACK_ERROR": {"employee_wrong_value": "READBACK_ERROR", "final_value": "FINAL"},
    "EC_ONE_FIELD_READBACK_ERROR": {"employee_wrong_value": "READBACK_ERROR", "final_value": "FINAL"},
    "EC_FALSE_START": {"final_value": "FINAL"},
    "EC_LONG_DISTANCE_DEPENDENCY": {},  # carries gap and target IDs, no product values
}
_ORDER_PARAMETERS = {"initial_order": "INITIAL_ORDER", "final_order": "FINAL"}
_REFERENCE_PARAMETERS = {"candidate_products": "CANDIDATE", "selected_product": "FINAL"}
_NON_PRODUCT_PARAMETERS = {"target_question_ids", "min_turn_gap", "antecedent_speaker"}


def _mentions(edge):
    """(role, product name) pairs of one assignment, in stored order; empty when it names no T01 product."""
    edge_id, params, targets = edge["edge_case_id"], edge["parameters"], edge["target_question_ids"]
    if edge_id == "EC_COMPLETE_ORDER_REPLACEMENT":  # T01 is part of both orders even when T01 itself does not change
        return [(role, params[key]["T01"]) for key, role in _ORDER_PARAMETERS.items() if key in params]
    if edge_id == "EC_PRODUCT_REFERENCE":
        mapping = _REFERENCE_PARAMETERS
    elif "T01" in targets:
        if edge_id not in _TARGETED_PARAMETERS:
            raise ValueError(f"{edge['assignment_id']}: edge case {edge_id} targets T01 but has no product mapping")
        mapping = _TARGETED_PARAMETERS[edge_id]
    else:
        return []
    unmapped = sorted(set(params) - set(mapping) - _NON_PRODUCT_PARAMETERS)
    if unmapped:
        raise ValueError(f"{edge['assignment_id']}: unmapped parameters {unmapped} on a T01-targeted {edge_id}")
    pairs = []
    for key, value in params.items():  # stored order
        if key not in mapping:
            continue
        pairs += [(mapping[key], name) for name in (value if isinstance(value, list) else [value])]
    return pairs


def derive_products(spec, product_category):
    """[{product_ref, name, category, roles}] in first-seen order for one call spec."""
    gold = {entry["question_id"]: entry["expected"] for entry in spec["gold"]}
    final_name = gold["T01"]["expected_value"]
    order, roles = [final_name], {final_name: {"FINAL"}}
    for edge in spec["edge_cases"]:
        for role, name in _mentions(edge):
            if name not in roles:
                order.append(name)
                roles[name] = set()
            roles[name].add(role)
            if role == "FINAL" and name != final_name:
                raise ValueError(f"{edge['assignment_id']}: final product {name!r} contradicts gold {final_name!r}")
    products = []
    for number, name in enumerate(order, start=1):
        if name not in product_category:
            raise ValueError(f"{spec['call_id']}: product {name!r} is not in the generation-plan product pool")
        products.append({
            "product_ref": f"P{number}", "name": name, "category": product_category[name],
            "roles": [role for role in PRODUCT_ROLES if role in roles[name]],
        })
    return products
