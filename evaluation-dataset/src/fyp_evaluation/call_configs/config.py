"""Version-controlled G3 configuration. The model ID lives here, never in `.env`."""

from pathlib import Path

from fyp_evaluation import release_versions as versions

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = PROJECT_ROOT / "prompts" / "stage3"
DEFAULT_AUDIT_ROOT = PROJECT_ROOT / ".llm_audit"

STAGE = "G3"
SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "1.0.0"
HASHING = "SHA-256 over UTF-8 bytes with CRLF normalised to LF"

# Approved request settings (identical for sync and batch).
MODEL = "gpt-6.1-sol"
REASONING_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 16000
RESPONSE_FORMAT_NAME = "g3_call_config_v1"
PROMPT_TEMPLATE = "call_config.v1.md"
RESPONSE_SCHEMA = "call_config_response"

# Generated paths, relative to the dataset root.
CONFIG_DIR = "authoring/call_configs"
PROVENANCE_DIR = "authoring/call_config_provenance"
RECORD = "authoring/call_config_record.json"

# Retry budgets per call, counted since the last accepted version.
SEMANTIC_BUDGET = 3  # REJECTED attempts (schema/rule failure, incomplete, refusal, unparseable)
TRANSPORT_BUDGET = 3  # TRANSPORT_FAILED attempts that are not fatal configuration errors

PRODUCT_ROLES = ("FINAL", "CANDIDATE", "INITIAL_ORDER", "SUPERSEDED", "READBACK_ERROR")


def schema_version(documents):
    """The artifact/record schema_version for a release: "2.0" for a multilingual release, else the historical "1.0"."""
    return versions.generation(documents)


def schema_names(documents):
    """(artifact schema, record schema) file stems; the provenance schema is shared by both generations."""
    return versions.versioned("call_config", documents), versions.versioned("call_config_record", documents)


def response_schema_name(documents):
    return versions.versioned(RESPONSE_SCHEMA, documents)


def prompt_template(documents):
    return "call_config.v3.md" if versions.generation(documents) == "2.0" else PROMPT_TEMPLATE


def response_format_name(documents):
    return versions.versioned_format(RESPONSE_FORMAT_NAME, documents)


def config_path(call_id):
    return f"{CONFIG_DIR}/{call_id}.json"


def provenance_path(call_id):
    return f"{PROVENANCE_DIR}/{call_id}.json"


def custom_id(call_id, attempt_id):
    """Valid for both OpenAI and Anthropic batch APIs: ^[a-zA-Z0-9_-]{1,64}$."""
    return f"{call_id}-{STAGE}-{attempt_id}"
