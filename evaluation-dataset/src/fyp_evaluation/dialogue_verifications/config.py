"""Version-controlled G7 configuration. The verifier model and its request settings live here, never in `.env`.

G7 uses the same model, tier, capacity policy and price table as G5; only the thinking level (high) and the limits differ.
"""

from pathlib import Path

from fyp_evaluation.plan_verifications.config import (  # noqa: F401  (shared with G5: same approved verifier and Flex tier)
    API_KEY_VARIABLE, ASSESSMENTS, CAPACITY_BACKOFF_SECONDS, CAPACITY_JITTER, CAPACITY_SENDS, MAX_EXPLANATION, MAX_QUOTE,
    MAX_RATIONALE, MODEL, PRICE_TABLE_VERSION, PRICES, QUESTION_IDS, SERVICE_TIER, TIMEOUT_MS, VERDICTS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = PROJECT_ROOT / "prompts" / "stage7"

STAGE = "G7"
SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "1.0.0"
HASHING = "SHA-256 over UTF-8 bytes with CRLF normalised to LF"

# Approved verifier settings: Gemini 3.8 Flash, HIGH thinking, paid Flex tier. There is never a fallback to Standard.
THINKING_LEVEL = "high"
MAX_OUTPUT_TOKENS = 32768  # thinking + report; a truncated report is REJECTED, never repaired

PROMPT_TEMPLATE = "dialogue_verification.v1.md"
RESPONSE_SCHEMA = "dialogue_verification_response"

ARTIFACT_DIR = "authoring/dialogue_verifications"
PROVENANCE_DIR = "authoring/dialogue_verification_provenance"
RECORD = "authoring/dialogue_verification_record.json"

SEMANTIC_BUDGET = 3  # structurally REJECTED reports per call since the last accepted version (never semantic verdicts)
TRANSPORT_BUDGET = 3

# Rough --dry-run planning figures (visible report + thinking), per duration profile. Estimates only.
ESTIMATED_OUTPUT_TOKENS = {"SHORT": 9000, "LONG": 18000}

ISSUE_CODES = ("COMPLIANCE_MEANING", "OMISSION_LEAKAGE", "EXTRACTION_STATE", "SCENARIO_CONTRADICTION", "EDGE_CASE",
               "CROSS_CHECK_INTERFERENCE", "MISSING_SUPPORT")
# A MISMATCH issue with any other code must cite the dialogue speech it rests on; MISSING_SUPPORT concerns something ABSENT.
UNGROUNDED_MISMATCH_CODE = "MISSING_SUPPORT"
REPAIR_SCOPES = ("G6_ONLY", "UPSTREAM_REVIEW", "REVIEW")
QUALITY_CODES = ("REPETITION", "ARTIFICIAL_EXTENSION", "INCOHERENT_FLOW", "IMPLAUSIBLE_TURN", "SPEAKER_ROLE")
SEVERITIES = ("NOTE", "SERIOUS")
QUALITY_REVIEWS = ("CLEAR", "REVIEW_REQUIRED")

MAX_EVIDENCE = 6
MAX_ISSUES = 20
MAX_QUALITY = 10
