"""Public validation API. Supply the schema directory explicitly."""

from .schemas import (
    check_schemas, load_json, to_langgraph_checklist, validate, validate_answers,
    validate_checklist,
)
from .dialogue import validate_speaker_alternation
from .evidence import resolve_evidence, validate_dialogue_against_plan
from .spoken import verify_spoken_value

# .transcript (validate_transcript_pair) is deliberately not re-exported here:
# it imports from fyp_evaluation.transcripts, which itself imports this
# package's .schemas. Re-exporting it from this __init__ would make that a
# package-init-time cycle. Import it directly from fyp_evaluation.validation.transcript
# (or from fyp_evaluation.transcripts, which re-exports it) instead.

__all__ = [
    "check_schemas", "load_json", "to_langgraph_checklist", "validate",
    "validate_answers", "validate_checklist",
    "resolve_evidence", "validate_dialogue_against_plan", "validate_speaker_alternation",
    "verify_spoken_value",
]
