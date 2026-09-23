"""Cross-file consistency checks between ground truth and its generated pair.

Structural shape of each individual document is checked by the shared JSON
Schema validator (`validate`, schema names "ground_truth", "transcript" and
"speaker_roles"). This module additionally checks that a generated
transcript and speaker-role mapping agree with each other and with the
ground truth they were built from, without recompiling them.
"""

from fyp_evaluation.transcripts.converter import SPEAKER_LABELS, SPEAKER_ROLES, TRANSCRIPT_SEPARATOR

from .schemas import validate


def validate_transcript_pair(ground_truth, transcript, speaker_roles, schema_dir):
    """Return a list of errors (empty if consistent); never raises or mutates.

    Checks structural validity of all three documents first, then: segment
    count/order/ids match dialogue turns, segment text is an exact copy,
    speaker-label mapping is correct, the full-transcript reconstruction
    matches, and every speaker label used in the transcript has exactly the
    expected semantic role in `speaker_roles` (no missing, no extra, no
    guessed role).
    """
    errors = [f"ground_truth: {error}" for error in validate(ground_truth, "ground_truth", schema_dir)]
    errors += [f"transcript: {error}" for error in validate(transcript, "transcript", schema_dir)]
    errors += [f"speaker_roles: {error}" for error in validate(speaker_roles, "speaker_roles", schema_dir)]
    if errors:
        return errors

    dialogue = ground_truth["dialogue"]
    segments = transcript["audio_segments"]
    if len(segments) != len(dialogue):
        return [f"audio_segments has {len(segments)} entries; dialogue has {len(dialogue)} turns"]

    for turn, segment in zip(dialogue, segments):
        turn_id = turn["turn_id"]
        if segment["id"] != turn_id:
            errors.append(f"turn {turn_id}: segment id {segment['id']!r} does not match turn_id")
        if segment["transcript"] != turn["text"]:
            errors.append(f"turn {turn_id}: segment transcript does not match dialogue text exactly")
        expected_label = SPEAKER_LABELS.get(turn["speaker"])
        if expected_label is None:
            errors.append(f"turn {turn_id}: no speaker-label mapping for speaker {turn['speaker']!r}")
        elif segment["speaker_label"] != expected_label:
            errors.append(
                f"turn {turn_id}: speaker_label {segment['speaker_label']!r} does not match "
                f"expected {expected_label!r} for speaker {turn['speaker']!r}"
            )

    expected_full = TRANSCRIPT_SEPARATOR.join(turn["text"] for turn in dialogue)
    actual_full = transcript["transcripts"][0]["transcript"]
    if actual_full != expected_full:
        errors.append("transcripts[0].transcript does not match the deterministic full-transcript reconstruction")

    used_labels = {segment["speaker_label"] for segment in segments}
    mapped_labels = set(speaker_roles)
    if used_labels != mapped_labels:
        errors.append(
            f"speaker_roles labels {sorted(mapped_labels)} do not match transcript labels {sorted(used_labels)}"
        )
    for label, role in speaker_roles.items():
        expected_role = SPEAKER_ROLES.get(label)
        if expected_role is not None and role != expected_role:
            errors.append(f"{label}: speaker role {role!r} does not match expected {expected_role!r}")
    return errors
