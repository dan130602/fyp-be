"""G8 inputs: the G7-verified G6 dialogue of every eligible call. Fails closed; an ineligible call is listed with its reason."""

from dataclasses import dataclass
from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text, text_hash
from fyp_evaluation.dialogues.inputs import read_dialogue

from .config import MAX_TURN_CHARACTERS, REPORT_DIR


@dataclass(frozen=True)
class RenderCall:
    call_id: str
    turns: tuple  # ({"turn_id", "speaker", "text"}, ...) exactly as in the accepted G6 dialogue
    dialogue_sha256: str
    report_sha256: str


def _turns(call_id, document):
    turns = document.get("turns")
    if document.get("call_id") != call_id or not isinstance(turns, list) or not turns:
        raise ValueError("the dialogue has no turns for this call")
    rows, speakers = [], ("EMPLOYEE", "CLIENT")
    first = speakers.index(turns[0].get("speaker")) if turns[0].get("speaker") in speakers else None
    if first is None:
        raise ValueError("the first turn has no EMPLOYEE/CLIENT speaker")
    for index, turn in enumerate(turns):
        text = turn.get("text")
        if turn.get("turn_id") != index or turn.get("speaker") != speakers[(first + index) % 2]:
            raise ValueError(f"turn {index} breaks the zero-based ID or strict speaker alternation")
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TURN_CHARACTERS:
            raise ValueError(f"turn {index} is empty or longer than {MAX_TURN_CHARACTERS} characters")
        rows.append({"turn_id": index, "speaker": turn["speaker"], "text": text})
    return tuple(rows)


def load_calls(root, schema_dir, call_ids=None):
    """(runner, calls, refused): `calls` maps call_id -> RenderCall for every selected call whose G7 report is current, intact and
    VALID; `refused` maps the others to their reason. Nothing is read from ground_truth.json."""
    from fyp_evaluation.dialogue_verifications.run import DialogueVerificationRunner

    root = Path(root)
    runner = DialogueVerificationRunner(root, schema_dir=schema_dir)
    selected = list(runner.inputs.call_ids if not call_ids else call_ids)
    unknown = [c for c in selected if c not in runner.inputs.call_ids]
    if unknown:
        raise ValueError(f"unknown call ids: {unknown}")
    calls, refused = {}, {}
    for call_id, state in runner.g8_eligibility(selected).items():
        if not state["eligible"]:
            refused[call_id] = state["reason"]
            continue
        try:
            fingerprint, text = read_dialogue(root, call_id)
            report = parse_json_text((root / REPORT_DIR / f"{call_id}.json").read_bytes().decode("utf-8").replace("\r\n", "\n"))
            if text is None or fingerprint is None or report.get("dialogue_sha256") != text_hash(text):
                raise ValueError("the G7 report was written for a different dialogue")
            calls[call_id] = RenderCall(call_id, _turns(call_id, parse_json_text(text)), text_hash(text),
                                        text_hash((root / REPORT_DIR / f"{call_id}.json").read_bytes().decode("utf-8").replace("\r\n", "\n")))
        except (OSError, ValueError, KeyError, TypeError) as error:
            refused[call_id] = f"cannot load the verified dialogue: {error}"[:300]
    return runner, calls, refused
