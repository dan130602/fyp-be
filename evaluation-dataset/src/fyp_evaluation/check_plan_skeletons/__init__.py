"""G2B: construct the deterministic check_plan_skeleton.json for every call.

The skeleton turns the immutable G2A call_spec plus the frozen G0 semantics, edge-case
catalog and normalization profiles into a purely structural evidence blueprint:
symbolic planned events, per-check evidence requirements, relative constraints and
provider-independent spoken-form requirements. It writes no dialogue wording, no turn
numbers and no SSML, and uses no LLM, network or AWS service.
"""
