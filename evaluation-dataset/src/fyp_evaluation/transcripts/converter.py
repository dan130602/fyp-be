"""Deterministic ground-truth-to-transcript and speaker-role conversion.

Transforms a canonical `ground_truth.json` into the two prepared documents
consumed by the offline NLP evaluation framework: a normalized AWS
Transcribe-shaped transcript (`transcripts` + `audio_segments`) and a
separate `speaker_roles` mapping. No LLM, Polly, AWS Transcribe, Bedrock or
other external/network call is made here; this module only rearranges
fields already present in the validated ground truth.

Only the explicit allowlist of fields below is ever read from ground truth.
Evaluation-only fields (`checks`, `expected`, `evidence_plan`,
`gold_evidence_turns`, `gold_evidence_text`, `critical_tokens`, ...) are
never copied into the generated documents.
"""

from pathlib import Path

from fyp_evaluation.validation.schemas import validate

_SCHEMAS = Path(__file__).resolve().parents[3] / "schemas"

# Fixed, order-independent mapping: which synthetic speaker ends up as which
# transcript speaker label and, in turn, which semantic role. EMPLOYEE is
# always spk_0/RM and CLIENT is always spk_1/Customer, regardless of which
# one speaks first in a given call's dialogue.
SPEAKER_LABELS = {"EMPLOYEE": "spk_0", "CLIENT": "spk_1"}
SPEAKER_ROLES = {"spk_0": "RM", "spk_1": "Customer"}

# The current NLP ingestion path reconstructs its transcript by joining
# diarized turns with two newlines; matching it here preserves compatibility
# and evidence character offsets once real timings replace these placeholders.
TRANSCRIPT_SEPARATOR = "\n\n"

# Audio generation (SSML/Polly/stitching) has not run yet for these calls, so
# there is no real audio-derived timing. These are temporary placeholders,
# not measurements, and must not be read as actual turn positions.
PLACEHOLDER_TIME = "0.0"


def build_transcript_artifacts(ground_truth, *, timings=None, schema_dir=_SCHEMAS):
    """Return (transcript, speaker_roles) built only from canonical dialogue.

    Pure and deterministic: never mutates `ground_truth`, performs no I/O and
    makes no external/network call. Byte-identical input always produces
    byte-identical output structures.

    `timings` is reserved for future normalized, audio-derived per-turn
    timing (once SSML/Polly rendering and stitching exist); accepting it
    here lets a future caller regenerate the same transcript shape with real
    timing without duplicating this mapping logic. It is not implemented
    yet, so any value other than None is rejected explicitly rather than
    silently ignored.
    """
    if timings is not None:
        raise NotImplementedError(
            "Timing enrichment is not implemented yet; call with timings=None "
            "until real audio-derived timings.json integration is authorized."
        )
    if errors := validate(ground_truth, "ground_truth", schema_dir):
        raise ValueError("\n".join(errors))

    dialogue = ground_truth["dialogue"]
    present_speakers = {turn["speaker"] for turn in dialogue}
    if unmapped := present_speakers - SPEAKER_LABELS.keys():
        raise ValueError(
            f"{ground_truth.get('call_id')}: no transcript speaker-label mapping for: {sorted(unmapped)}"
        )

    audio_segments = [
        {
            "id": turn["turn_id"],
            "transcript": turn["text"],
            "start_time": PLACEHOLDER_TIME,
            "end_time": PLACEHOLDER_TIME,
            "speaker_label": SPEAKER_LABELS[turn["speaker"]],
        }
        for turn in dialogue
    ]
    full_transcript = TRANSCRIPT_SEPARATOR.join(turn["text"] for turn in dialogue)
    transcript = {
        "transcripts": [{"transcript": full_transcript}],
        "audio_segments": audio_segments,
    }
    # Fixed EMPLOYEE-then-CLIENT iteration keeps spk_0 before spk_1 in the
    # output regardless of which speaker's turn comes first in the dialogue.
    speaker_roles = {
        "speaker_roles": {
            SPEAKER_LABELS[speaker]: SPEAKER_ROLES[SPEAKER_LABELS[speaker]]
            for speaker in ("EMPLOYEE", "CLIENT")
            if speaker in present_speakers
        }
    }

    if errors := validate(transcript, "transcript", schema_dir):
        raise ValueError("\n".join(errors))
    if errors := validate(speaker_roles, "speaker_roles", schema_dir):
        raise ValueError("\n".join(errors))
    return transcript, speaker_roles
