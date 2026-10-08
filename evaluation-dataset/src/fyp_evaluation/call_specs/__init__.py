"""G2A: compile the immutable per-call call_spec.json from the frozen G1 outputs.

G2A is a deterministic join of call_manifest.csv, coverage_matrix.csv and
edge_case_assignments.csv. It samples, rebalances, normalizes and invents
nothing, and uses no LLM, network or AWS service.
"""
