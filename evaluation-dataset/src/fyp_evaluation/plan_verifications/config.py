"""Version-controlled G5 configuration. The verifier model and its request settings live here, never in `.env`."""

from pathlib import Path

from fyp_evaluation import release_versions as versions

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = PROJECT_ROOT / "prompts" / "stage5"

STAGE = "G5"
SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "1.0.0"
HASHING = "SHA-256 over UTF-8 bytes with CRLF normalised to LF"

# Approved verifier settings: Gemini 3.8 Flash, medium thinking, paid Flex tier. There is never a fallback to Standard.
MODEL = "gemini-3.8-flash"
THINKING_LEVEL = "medium"
SERVICE_TIER = "flex"
MAX_OUTPUT_TOKENS = 32768  # thinking + report; a truncated report is REJECTED, never repaired
API_KEY_VARIABLE = "GEMINI_API_KEY"
TIMEOUT_MS = 900_000  # Flex requests can queue; Google recommends client timeouts of 10 minutes or more

PROMPT_TEMPLATE = "plan_verification.v1.md"
RESPONSE_SCHEMA = "plan_verification_response"

def schema_version(documents):
    """The artifact/record schema_version for a release: "2.0" for a multilingual release, else the historical "1.0"."""
    return versions.generation(documents)


def schema_names(documents):
    """(artifact schema, record schema) file stems; the provenance schema is shared by both generations."""
    return versions.versioned("plan_verification", documents), versions.versioned("plan_verification_record", documents)


def response_schema_name(documents):
    return versions.versioned(RESPONSE_SCHEMA, documents)


def prompt_template(documents):
    return versions.versioned_prompt(PROMPT_TEMPLATE, documents)


ARTIFACT_DIR = "authoring/plan_verifications"
PROVENANCE_DIR = "authoring/plan_verification_provenance"
RECORD = "authoring/plan_verification_record.json"

SEMANTIC_BUDGET = 3  # structurally REJECTED reports per call since the last accepted version (never semantic verdicts)
TRANSPORT_BUDGET = 3

# Capacity (429/503) is retried inside one attempt with identical request bytes; the application is the only retry owner.
CAPACITY_SENDS = 4
CAPACITY_BACKOFF_SECONDS = (30.0, 90.0, 270.0)
CAPACITY_JITTER = 0.2  # +/- fraction applied to each backoff

# Dated price table (USD per million tokens). Estimates only; thinking tokens are billed as output.
PRICE_TABLE_VERSION = "2026-10-11"
PRICES = {MODEL: {"flex": {"input": 0.375, "cached_input": 0.0375, "output": 1.875}}}
ESTIMATED_OUTPUT_TOKENS = 8000  # report + thinking, for --dry-run planning only

QUESTION_IDS = ("C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12",
                "K01", "K02", "T01", "T02", "T03", "T04", "T05", "T06")
ASSESSMENTS = ("MATCH", "MISMATCH", "UNCLEAR")
ISSUE_CODES = ("SCENARIO_CONTRADICTION", "COMPLIANCE_MEANING", "OMISSION_LEAKAGE", "EXTRACTION_STATE", "EDGE_CASE",
               "PLAN_COHERENCE")
ISSUE_CODES_V2 = ISSUE_CODES + ("LANGUAGE_REQUIREMENT_NOT_MET",)  # the multilingual (schema 2.0) generation only
REPAIR_SCOPES = ("G4_ONLY", "G3_PLUS_G4", "REVIEW")
VERDICTS = ("VALID", "INVALID", "UNCLEAR")

MAX_RATIONALE = 700
MAX_EXPLANATION = 1200
MAX_QUOTE = 400
MAX_EVIDENCE = 6
MAX_ISSUES = 20
