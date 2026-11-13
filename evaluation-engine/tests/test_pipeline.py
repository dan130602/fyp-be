"""The one-command evaluation (scripts/run_eval.py): which existing scripts run, in what order, with which ids."""
import pytest

from transcribe_eval.pipeline import PipelineError, final_run_id, plan_steps


def scripts(steps):
    return [argv[2].split("/")[-1] for _, argv in steps]


def test_a_clean_run_is_collect_score_export_over_every_call():
    steps = plan_steps("dev-clean-2", python="py")
    assert scripts(steps) == ["collect.py", "score.py", "export_to_fyp_ai.py"]
    collect = steps[0][1]
    assert collect[:2] == ["py", "-B"] and "--all" in collect and "--audio-condition" not in collect and collect[collect.index("--run-id") + 1] == "dev-clean-2"
    assert steps[1][1][-2:] == ["--run", "dev-clean-2"] and steps[2][1][-2:] == ["--run", "dev-clean-2"]


def test_a_noisy_run_that_sends_every_call_has_no_assemble_step_and_compares_against_the_clean_run():
    steps = plan_steps("dev-v2-1", audio_condition="environment_v2", clean_run="dev175-multilang-1", python="py")
    assert scripts(steps) == ["collect.py", "score.py", "compare_conditions.py", "export_to_fyp_ai.py"]
    collect = steps[0][1]
    assert collect[collect.index("--audio-condition") + 1] == "environment_v2" and "--skip-clean-controls" not in collect
    assert steps[2][1][-4:] == ["--clean-run", "dev175-multilang-1", "--v2-run", "dev-v2-1"]


def test_skipping_the_clean_calls_adds_the_assemble_step_and_everything_after_it_uses_the_assembled_run():
    steps = plan_steps("dev-v2-1", audio_condition="environment_v2", skip_clean_controls=True, clean_run="dev175-multilang-1", python="py")
    assert scripts(steps) == ["collect.py", "assemble_v2_run.py", "score.py", "compare_conditions.py", "export_to_fyp_ai.py"]
    assert "--skip-clean-controls" in steps[0][1]
    assert steps[1][1][-6:] == ["--v2-run", "dev-v2-1", "--clean-run", "dev175-multilang-1", "--out", "dev-v2-1-full"]
    assert all(argv[-1] == "dev-v2-1-full" for _, argv in steps[2:])
    assert final_run_id("dev-v2-1", "environment_v2", True) == "dev-v2-1-full" and final_run_id("dev-v2-1", "environment_v2", False) == "dev-v2-1"


def test_a_canary_names_its_calls_and_publishing_is_never_a_step():
    steps = plan_steps("dev-v2-canary", audio_condition="environment_v2", clean_run="c", calls=["CALL_033", "CALL_064"], python="py")
    collect = steps[0][1]
    assert collect[collect.index("--calls") + 1:collect.index("--calls") + 3] == ["CALL_033", "CALL_064"] and "--all" not in collect
    assert not any("upload" in " ".join(argv) for _, argv in steps)


def test_compare_and_export_can_be_turned_off():
    steps = plan_steps("r", audio_condition="environment_v2", compare=False, export=False, python="py")
    assert scripts(steps) == ["collect.py", "score.py"]


@pytest.mark.parametrize("kwargs", [dict(audio_condition="environment_v2"), dict(audio_condition="environment_v2", skip_clean_controls=True, compare=False),
                                    dict(skip_clean_controls=True), dict(audio_condition="noisy")])
def test_combinations_that_cannot_work_are_refused_before_anything_is_paid_for(kwargs):
    with pytest.raises(PipelineError):
        plan_steps("r", **kwargs)
