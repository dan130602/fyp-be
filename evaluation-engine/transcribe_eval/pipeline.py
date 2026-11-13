"""The steps of one evaluation, in order, as the existing scripts' command lines (so run_eval.py adds no logic of its own).

Clean:   collect -> score -> export
Noisy:   collect -> [assemble, only when the clean calls were skipped] -> score -> compare -> export
Publishing to Supabase is never a step: it is immutable, so it stays a deliberate command (see publish_hint).
"""
import sys

DEFAULT_DATASET = "advisory_execution_multilingual_v1_development"
V2 = "environment_v2"


class PipelineError(ValueError):
    pass


def final_run_id(run_id, audio_condition, skip_clean_controls):
    """The run that is scored, compared and exported: the assembled run when the clean calls were skipped, otherwise the collected run."""
    return f"{run_id}-full" if audio_condition == V2 and skip_clean_controls else run_id


def plan_steps(run_id, *, dataset=DEFAULT_DATASET, audio_condition="clean", skip_clean_controls=False, clean_run=None, calls=None,
               timeout_minutes=120, compare=True, export=True, python=None):
    """[(label, argv)] for the whole evaluation; raises PipelineError for a combination that cannot work."""
    if audio_condition not in ("clean", V2):
        raise PipelineError(f"unknown audio condition {audio_condition!r}")
    if skip_clean_controls and audio_condition != V2:
        raise PipelineError("--skip-clean-controls only applies to --audio-condition environment_v2")
    if audio_condition == V2 and (skip_clean_controls or compare) and not clean_run:
        raise PipelineError("an environment_v2 run needs --clean-run <an earlier scored clean run> (for the paired comparison"
                            + (" and for the clean calls it does not transcribe again)" if skip_clean_controls else ")") + "; or pass --no-compare")
    python = python or sys.executable
    run = [python, "-B"]
    collect = run + ["scripts/collect.py", "--dataset", dataset, "--render-id", "auto", "--run-id", run_id, "--timeout-minutes", str(timeout_minutes)]
    collect += ["--calls", *calls] if calls else ["--all"]
    if audio_condition == V2:
        collect += ["--audio-condition", V2] + (["--skip-clean-controls"] if skip_clean_controls else [])
    steps = [("collect (Amazon Transcribe, billable)", collect)]
    final = final_run_id(run_id, audio_condition, skip_clean_controls)
    if final != run_id:
        steps.append(("assemble (adds the earlier clean transcripts, offline)",
                      run + ["scripts/assemble_v2_run.py", "--v2-run", run_id, "--clean-run", clean_run, "--out", final]))
    steps.append(("score (offline)", run + ["scripts/score.py", "--run", final]))
    if audio_condition == V2 and compare:
        steps.append(("compare clean versus changed (offline)", run + ["scripts/compare_conditions.py", "--clean-run", clean_run, "--v2-run", final]))
    if export:
        steps.append(("export for the fyp-ai dashboard (offline)", run + ["scripts/export_to_fyp_ai.py", "--run", final]))
    return steps


def publish_hint(final):
    return f"To publish to Supabase (immutable, so deliberate), from the fyp-ai folder: python -B evaluation/scripts/upload_stt_run.py --run-id {final}"
