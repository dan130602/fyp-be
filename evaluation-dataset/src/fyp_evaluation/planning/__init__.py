"""G1 deterministic dataset planning for versioned dataset releases.

G1 reads the approved G0 (Stage 0) contracts and writes call_manifest.csv,
coverage_matrix.csv, edge_case_assignments.csv and planning_record.json. It
uses no LLM, network or AWS service and never reads generated dialogue.
"""
