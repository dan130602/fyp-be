"""Version-controlled G4 configuration. The model ID is shared with G3 (same approved authoring model)."""

from pathlib import Path

from fyp_evaluation.call_configs.config import MAX_OUTPUT_TOKENS, MODEL, REASONING_EFFORT  # noqa: F401  (re-exported)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = PROJECT_ROOT / "prompts" / "stage4"

STAGE = "G4"
SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "1.0.0"
HASHING = "SHA-256 over UTF-8 bytes with CRLF normalised to LF"

RESPONSE_FORMAT_NAME = "g4_check_plan_v1"
PROMPT_TEMPLATE = "check_plan.v1.md"
RESPONSE_SCHEMA = "check_plan_response"

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
