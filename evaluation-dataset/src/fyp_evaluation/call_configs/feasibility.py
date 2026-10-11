"""Deterministic synthetic responses and the all-calls feasibility pass.

`synthesize_response` builds a response that satisfies exactly the required slots of a call and nothing else. It
is used offline (tests, `--dry-run`) to prove that every frozen call has at least one valid response before any
paid request is made. It is never sent to a provider and never written as an artifact.
"""

from .compile import compile_artifact
from .validate_response import ResponseValidator


def synthesize_response(ctx, *, fee_basis="RATE"):
    products = {p["product_ref"]: p for p in ctx.products}
    response = {
        "client": {"name": "Alex Tan", "background": "A long-standing private banking client with a diversified portfolio.",
                   "investment_objective": "Long-term capital growth with moderate risk.",
                   "loss_attitude": "Prefers to limit losses but understands markets can fall."},
        "employee": {"name": "Jamie Lim"},
        "setting": {"call_reason": "The client phones to act on a portfolio review.", "conversation_style": "Courteous and efficient."},
        "final_product": {"description": f"A {ctx.products[0]['category'].lower().replace('_', ' ')} offered through the bank.",
                          "concrete_risks": ["The unit price can fall with market movements."],
                          "recommendation_basis": "Its features suit the client's stated need for steady growth."},
        "other_products": [{"product_ref": ref, "description": f"Another {products[ref]['category'].lower().replace('_', ' ')} mentioned in passing."}
                           for ref in ctx.other_product_refs],
        "holding_context": "The client already holds this product in the portfolio." if ctx.holding_required else None,
        "fee_disclosure": None,
        "limit_prices": {key: (_limit_price(ctx.order_states[key]) if key in ctx.order_states and ctx.order_states[key]["limit_required"] else None)
                         for key in ("final", "initial")},
        "red_flags": {q: (f"A short factual description for check {q}." if required else None) for q, required in ctx.red_flag_required.items()},
        "edge_case_context": [{"assignment_id": aid, "motivation": "The client reconsiders after thinking the matter over."}
                              for aid in ctx.edge_context_ids],
        "additional_discussion_topics": ["General market sentiment", "Recent portfolio review"],
    }
    if ctx.fee_required:
        if fee_basis == "AMOUNT":
            response["fee_disclosure"] = {"basis": "AMOUNT", "value": 37.5, "currency": ctx.fee_currencies[0]}
        else:
            response["fee_disclosure"] = {"basis": "RATE", "value": 0.75, "currency": None}
    return response


def _limit_price(state):
    return round(state["implied_unit_price"], 2)


def feasibility_report(builder, validator, inputs):
    """{call_id: [errors]} for the synthetic response of every call (empty lists mean feasible)."""
    report = {}
    for call_id in inputs.call_ids:
        ctx = builder.context(call_id)
        response = synthesize_response(ctx)
        errors = validator.errors(response, ctx)
        if not errors:
            generation = {"attempt_id": "A001", "model": "fixture", "prompt_template_sha256": "0" * 64,
                          "response_schema_sha256": builder.response_schema_sha256}
            compile_artifact(inputs, ctx, response, generation, builder.response_schema)
        report[call_id] = errors
    return report


def make_validator(builder):
    return ResponseValidator(builder.response_schema)
