"""Version-controlled G6 configuration. The model ID is shared with G3/G4 (same approved authoring model)."""

from pathlib import Path

from fyp_evaluation import release_versions as versions
from fyp_evaluation.call_configs.config import MODEL, REASONING_EFFORT  # noqa: F401  (re-exported)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = PROJECT_ROOT / "prompts" / "stage6"

STAGE = "G6"
SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "1.0.0"
HASHING = "SHA-256 over UTF-8 bytes with CRLF normalised to LF"

# Synchronous G6 generation only: a long reasoning request may take many minutes, so the HTTP timeout is generous, and the SDK never
# retries it (a retry would silently pay for a second dialogue). A timeout is recorded as an ambiguous, possibly billed attempt.
SYNC_TIMEOUT_SECONDS = 1800.0

# One cap for both duration profiles (reasoning tokens included); a truncated dialogue is REJECTED, never repaired.
MAX_OUTPUT_TOKENS = 32768
RESPONSE_FORMAT_NAME = "g6_dialogue_v1"
PROMPT_TEMPLATE = "dialogue.v1.md"
RESPONSE_SCHEMA = "dialogue_response"

ARTIFACT_DIR = "authoring/dialogues"
PROVENANCE_DIR = "authoring/dialogue_provenance"
RECORD = "authoring/dialogue_record.json"

G5_ARTIFACT_DIR = "authoring/plan_verifications"
G5_RECORD = "authoring/plan_verification_record.json"

SEMANTIC_BUDGET = 3  # structurally REJECTED attempts per call since the last accepted version
TRANSPORT_BUDGET = 3

# Words -> estimated audio duration. Calibrated on the repository's own Polly renders (legacy_pilot_v1, clean_voice_A:
# 150-169 words/minute including inter-turn gaps). Each accepted band keeps the frozen duration profile for any rate from
# 150 to 170 wpm. Recorded on every attempt, so a historical dialogue is always rebuilt with the policy it was made under.
LENGTH_POLICY = {
    "version": "g6-length-v1",
    "assumed_wpm": 160,
    "profiles": {
        "SHORT": {"accepted_words": [510, 750], "target_words": [580, 680]},
        "LONG": {"accepted_words": [5103, 9000], "target_words": [5600, 6600]},
    },
}

# Multilingual releases count length in the unit of the call's language profile. ENGLISH and SINGLISH keep the historical policy
# unchanged (words); MANDARIN counts Han characters; EN_ZH_MIXED counts Han characters plus English words. The bands keep the
# frozen duration profiles (SHORT 180-300 s, LONG 1800-3600 s) over a stated rate range: accepted = duration bound x the rate
# range's edge. ALL NON-WORD RATES AND BANDS ARE PROVISIONAL until the live pilot's audio calibrates them. The policy keeps the
# provenance shape (`assumed_wpm` is the assumed units per minute, `*_words` the unit counts); its version names the unit.
UNIT_OF_POLICY = {"g6-length-v1": "words", "g6-length-v2-han-chars": "han_chars", "g6-length-v2-han-chars-plus-words": "han_chars_plus_words"}
# Advisory versions retain the same targets and counting rules. Old versions stay blocking when history is rebuilt.
ADVISORY_POLICIES = {version: version + "-advisory-v1" for version in UNIT_OF_POLICY}
UNIT_OF_POLICY.update({ADVISORY_POLICIES[version]: unit for version, unit in list(UNIT_OF_POLICY.items())})
LENGTH_POLICY_V2 = {
    "MANDARIN": {
        "version": "g6-length-v2-han-chars",
        "assumed_wpm": 260,  # Han characters per minute, inter-turn gaps included (provisional; accepted for 235-285)
        "profiles": {"SHORT": {"accepted_words": [855, 1175], "target_words": [945, 1105]},
                     "LONG": {"accepted_words": [8550, 14100], "target_words": [9100, 10725]}},
    },
    "EN_ZH_MIXED": {
        "version": "g6-length-v2-han-chars-plus-words",
        "assumed_wpm": 215,  # Han characters plus English words per minute (provisional; accepted for 190-240)
        "profiles": {"SHORT": {"accepted_words": [720, 950], "target_words": [780, 915]},
                     "LONG": {"accepted_words": [7200, 11400], "target_words": [7525, 8870]}},
    },
}


# Multilingual attempts record the placement-rule set they were written and validated under (`params.placement_rules`). Entries that
# predate a rule (no key) are re-validated without it, so history stays valid under the rules in force when it was accepted.
# g6-placement-v2: adds the derived order of cross-language distractor variants (language_order.py).
PLACEMENT_RULES = "g6-placement-v2"


def schema_version(documents):
    return versions.generation(documents)


def schema_names(documents):
    """(artifact schema, record schema) file stems; the provenance schema is shared by both generations."""
    return versions.versioned("dialogue", documents), versions.versioned("dialogue_record", documents)


def response_schema_name(documents):
    return versions.versioned(RESPONSE_SCHEMA, documents)


# New multilingual requests use v5 (v4's advisory length plus whole-call naturalness: topic development, role, chronology, opening,
# turn length, code-switching). Earlier prompt files and the policies recorded in history stay intact.
PROMPT_TEMPLATE_MULTILINGUAL = "dialogue.v5.md"


def prompt_template(documents):
    if versions.generation(documents) == versions.MULTILINGUAL:
        return PROMPT_TEMPLATE_MULTILINGUAL
    return PROMPT_TEMPLATE


def response_format_name(documents):
    return versions.versioned_format(RESPONSE_FORMAT_NAME, documents)


def length_policy_for(spec, documents):
    """Historical releases keep blocking length; new multilingual requests record an advisory version of their profile's policy."""
    if versions.generation(documents) == versions.HISTORICAL:
        return LENGTH_POLICY
    policy = LENGTH_POLICY_V2.get(spec["metadata"].get("language_profile"), LENGTH_POLICY)
    return {**policy, "version": ADVISORY_POLICIES[policy["version"]]}


def length_is_advisory(policy):
    return policy["version"] in ADVISORY_POLICIES.values()


def unit_of(policy):
    return UNIT_OF_POLICY[policy["version"]]


# A G5 report that exists but cannot be read: its fingerprint cannot be established (never confused with deletion).
G5_UNREADABLE = "UNREADABLE"
# The same for a G6 dialogue as seen by G7: a file that exists but cannot be read is never classified as deleted.
DIALOGUE_UNREADABLE = "UNREADABLE"

IDENTITY_QUESTIONS = ("K01", "K02")
EVENT_RELATIONS = ("corrects_event_id", "refers_to_event_id", "invalidates_event_id")

# Rough --dry-run planning figures (visible output + reasoning), per duration profile.
ESTIMATED_OUTPUT_TOKENS = {"SHORT": 6000, "LONG": 24000}
