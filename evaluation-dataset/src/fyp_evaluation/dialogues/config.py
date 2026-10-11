"""Version-controlled G6 configuration. The model ID is shared with G3/G4 (same approved authoring model)."""

from pathlib import Path

from fyp_evaluation.call_configs.config import MODEL, REASONING_EFFORT  # noqa: F401  (re-exported)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = PROJECT_ROOT / "prompts" / "stage6"

STAGE = "G6"
SCHEMA_VERSION = "1.0"
GENERATOR_VERSION = "1.0.0"
HASHING = "SHA-256 over UTF-8 bytes with CRLF normalised to LF"

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

# A G5 report that exists but cannot be read: its fingerprint cannot be established (never confused with deletion).
G5_UNREADABLE = "UNREADABLE"

IDENTITY_QUESTIONS = ("K01", "K02")
EVENT_RELATIONS = ("corrects_event_id", "refers_to_event_id", "invalidates_event_id")

# Rough --dry-run planning figures (visible output + reasoning), per duration profile.
ESTIMATED_OUTPUT_TOKENS = {"SHORT": 6000, "LONG": 24000}
