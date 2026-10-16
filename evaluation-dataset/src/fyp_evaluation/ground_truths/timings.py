"""Audio-derived timings for the FINAL G9 compile (offline; the G8 render must already exist).

The final compile stitches the per-turn times of a G8 render into `transcript.json` and records where they came from. Every call must
pass the G8 gate first: its G7 report is current, intact and VALID, its render is bound to the CURRENT accepted dialogue and G7 report,
its `timings.json` covers every dialogue turn exactly once, and (when present) its audio matches the recorded hash. Nothing here reads or
changes gold, the dialogue or the render.
"""

from pathlib import Path

from fyp_evaluation.call_specs.compile import parse_json_text, text_hash
from fyp_evaluation.rendering.inputs import load_calls
from fyp_evaluation.rendering.render import RENDER_MANIFEST, TIMINGS, check_render, render_dir

RECORD = "authoring/ground_truth_record.json"


def recorded_render_id(root):
    """The render a FINAL record was compiled from, or None for a provisional (or absent) record. Never raises."""
    try:
        record = parse_json_text((Path(root) / RECORD).read_bytes().decode("utf-8").replace("\r\n", "\n"))
        return record["timing"]["render_id"] if record["status"]["timings"] == "AUDIO_DERIVED" else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_timings(root, schema_dir, render_id):
    """{call_id: {"turns": [...], "provenance": {...}}} for every call, or ValueError listing what blocks the final compile."""
    root = Path(root)
    _, calls, refused = load_calls(root, schema_dir)
    problems = [f"{call_id}: not G8-eligible ({reason})" for call_id, reason in sorted(refused.items())]
    result = {}
    for call_id, call in calls.items():
        errors, _ = check_render(root, call, render_id, schema_dir)
        if errors:
            problems += [f"{call_id}: render {render_id}: {error}" for error in errors[:3]]
            continue
        directory = render_dir(root, call_id, render_id)
        timings_text = (directory / TIMINGS).read_bytes().decode("utf-8").replace("\r\n", "\n")
        manifest_text = (directory / RENDER_MANIFEST).read_bytes().decode("utf-8").replace("\r\n", "\n")
        timings, manifest = parse_json_text(timings_text), parse_json_text(manifest_text)
        result[call_id] = {
            "turns": timings["turns"],
            "provenance": {"render_id": render_id, "timings_sha256": text_hash(timings_text), "render_manifest_sha256": text_hash(manifest_text),
                           "audio_sha256": manifest["audio_sha256"], "g7_report_sha256": call.report_sha256,
                           "duration_seconds": timings["duration_seconds"]}}
    if problems:
        raise ValueError(f"the final G9 compile needs a complete, current G8 render ({render_id}):\n" + "\n".join(problems[:20]))
    return result
