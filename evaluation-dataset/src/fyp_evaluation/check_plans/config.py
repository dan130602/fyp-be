"""Version-controlled G4 configuration. The model ID is shared with G3 (same approved authoring model)."""

from pathlib import Path

from fyp_evaluation.call_configs.config import MAX_OUTPUT_TOKENS, MODEL, REASONING_EFFORT  # noqa: F401  (re-exported)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = PROJECT_ROOT / "prompts" / "stage4"

STAGE = "G4"
SCHEMA_VERSION = "1.0"
MULTILINGUAL_SCHEMA_VERSION = "2.0"  # plans and records of a multilingual (schema-2.0) release
GENERATOR_VERSION = "1.0.0"
HASHING = "SHA-256 over UTF-8 bytes with CRLF normalised to LF"

RESPONSE_FORMAT_NAME = "g4_check_plan_v1"
PROMPT_TEMPLATE = "check_plan.v1.md"
RESPONSE_SCHEMA = "check_plan_response"

def schema_version(documents):
    """The artifact/record schema_version for a release: "2.0" for a multilingual release, else the historical "1.0"."""
    from fyp_evaluation.validation.stage0 import is_multilingual

    return MULTILINGUAL_SCHEMA_VERSION if is_multilingual(documents) else SCHEMA_VERSION


def schema_names(documents):
    """(artifact schema, record schema) file stems for a release; the 2.0 files add `language` to mentions and the language-profile hash."""
    if schema_version(documents) == SCHEMA_VERSION:
        return "check_plan", "check_plan_record"
    return "check_plan.v2", "check_plan_record.v2"


ARTIFACT_DIR = "authoring/check_plans"
PROVENANCE_DIR = "authoring/check_plan_provenance"
RECORD = "authoring/check_plan_record.json"

SEMANTIC_BUDGET = 3  # structurally REJECTED attempts per call since the last accepted version
TRANSPORT_BUDGET = 3

# Closed placeholder vocabulary (model-authored templates may only use these, `[[NAME]]`).
QUESTION_PLACEHOLDERS = ("K01", "K02", "T01", "T02", "T03", "T04", "T05", "T06")
SCENARIO_PLACEHOLDERS = ("FEE", "LIMIT_PRICE", "CURRENCY")
ALL_PLACEHOLDERS = QUESTION_PLACEHOLDERS + SCENARIO_PLACEHOLDERS

MAX_TEMPLATE_CHARACTERS = 400
