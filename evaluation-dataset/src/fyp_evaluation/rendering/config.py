"""G8 constants: SSML conversion, Amazon Polly rendering and per-turn timings of the G7-verified dialogues."""

from dataclasses import dataclass

STAGE, SCHEMA_VERSION = "G8", "1.0"
DIALOGUE_DIR = "authoring/dialogues"
REPORT_DIR = "authoring/dialogue_verifications"
CALLS_DIR, RENDERS_DIR = "calls", "renders"
DEFAULT_RENDER_ID = "clean_voice_A"

SAMPLE_RATE = 16000  # Polly PCM is signed, little-endian, 16-bit mono
BYTES_PER_SECOND = SAMPLE_RATE * 2
MAX_TURN_CHARACTERS = 3000  # Polly's per-request limit on billed characters

# Pronunciation hints live in the render input only; the canonical dialogue text is never changed (README: reference orthography).
PRONUNCIATION_VERSION = "ssml-v3"

# Approximate public list price per million billed characters, for the cost PREVIEW only (verify against the AWS price list; the
# free tier and regional differences are not modelled).
APPROX_USD_PER_MILLION_CHARACTERS = {"neural": 16.0, "standard": 4.0}


@dataclass(frozen=True)
class RenderSettings:
    region: str = "ap-southeast-1"
    engine: str = "neural"
    employee_voice: str = "Jasmine"
    client_voice: str = "Matthew"
    silence_ms: int = 400

    def voice(self, speaker):
        return {"EMPLOYEE": self.employee_voice, "CLIENT": self.client_voice}[speaker]

    def as_record(self):
        return {"region": self.region, "engine": self.engine, "employee_voice": self.employee_voice, "client_voice": self.client_voice,
                "silence_ms": self.silence_ms, "sample_rate": SAMPLE_RATE, "pronunciation": PRONUNCIATION_VERSION}
