"""Structural dialogue rules that the JSON schema cannot express."""


def validate_speaker_alternation(dialogue):
    """Return one error per run of consecutive turns by the same speaker.

    A turn is one uninterrupted stretch of speech, so a speaker never has two
    turns in a row. A speaker who continues speaking must be merged into a
    single turn, and turns must alternate EMPLOYEE/CLIENT. Runs are reported by
    zero-based position in ``dialogue["turns"]``, which is also the future
    canonical turn ID because IDs are assigned by position at compilation.
    """
    errors = []
    turns = dialogue["turns"]
    start = 0
    for index in range(1, len(turns) + 1):
        if index < len(turns) and turns[index]["speaker"] == turns[start]["speaker"]:
            continue
        if index - start > 1:
            errors.append(
                f"turns {start}-{index - 1}: {index - start} consecutive {turns[start]['speaker']} turns; "
                "merge them into one turn (speakers must alternate)"
            )
        start = index
    return errors
