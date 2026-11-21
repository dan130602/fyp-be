"""D4: the final text reaches the adapter's own field, and the written document carries everything the plan lists."""
import json

import pytest

import synthetic
from harness import raw_files, run, two_jobs
from meralion_worker import config as cfg
from meralion_worker import output
from meralion_worker.state import sha256_bytes


def read_output(tmp_path, name, base="run"):
    data = (tmp_path / base / "out" / "transcripts" / f"{name}.json").read_bytes()
    return json.loads(data.decode("utf-8")), data


def test_the_formatted_final_text_reaches_results_transcripts(tmp_path):
    world = two_jobs()
    run(world, tmp_path)
    document, _ = read_output(tmp_path, "beta")
    transcript = document["results"]["transcripts"][0]["transcript"]
    assert "0.5 percent" in transcript and "half a percent" not in transcript
    block = document["meralion"]
    assert "half a percent" in block["transcript_raw"] and block["transcript_final"] == transcript
    assert block["formatter"] == {"name": "percentages-v2+amount-v1", "version": "2+amount-v1", "changed": True}          # the default (amt1) configuration
    assert set(document["results"]) == {"transcripts"} and document["jobName"] == "beta" and document["status"] == "COMPLETED"


def test_a_recording_without_a_formatter_target_is_unchanged(tmp_path):
    world = two_jobs()
    run(world, tmp_path)
    document, _ = read_output(tmp_path, "alpha")
    assert document["meralion"]["formatter"]["changed"] is False
    assert document["meralion"]["transcript_raw"] == document["meralion"]["transcript_final"] == document["results"]["transcripts"][0]["transcript"]


def test_formatter_none_leaves_raw_equal_to_final(tmp_path):
    world = two_jobs()
    run(world, tmp_path, effective=synthetic.effective(formatter="none"))
    document, _ = read_output(tmp_path, "beta")
    block = document["meralion"]
    assert "half a percent" in document["results"]["transcripts"][0]["transcript"] == block["transcript_raw"] == block["transcript_final"]
    assert block["formatter"] == {"name": "none", "version": None, "changed": False}


def test_the_document_validates_and_round_trips_through_the_serialiser(tmp_path):
    run(two_jobs(), tmp_path)
    for name in ("alpha", "beta"):
        document, data = read_output(tmp_path, name)
        output.validate_document(document)
        assert output.serialize(document) == data


def test_the_invocation_block_ties_the_output_to_the_plan(tmp_path):
    world = two_jobs()
    run(world, tmp_path)
    document, _ = read_output(tmp_path, "alpha")
    plan_bytes = (tmp_path / "run" / "state" / "inv-1" / "plan.json").read_bytes()
    assert document["meralion"]["invocation"] == {"id": "inv-1", "members": 2, "windows_total": 14, "run_budget": None, "run_budget_bound": False, "plan_sha256": sha256_bytes(plan_bytes)}                  # the default strategy has no invocation-wide budget


def test_the_config_fingerprint_decoder_model_and_worker_blocks_are_recorded(tmp_path):
    run(two_jobs(), tmp_path)
    document, _ = read_output(tmp_path, "alpha")
    block, effective = document["meralion"], synthetic.effective()
    assert block["config_fingerprint"] == effective.fingerprint and block["config"] == effective.data
    assert block["decoder"]["backend"] == "replay" and block["model"] == synthetic.context().model
    assert block["worker"]["version"] == "0.1.0" and block["schema_version"] == "meralion-worker-output-v0"


def test_the_audio_block_comes_from_the_recording(tmp_path):
    world = two_jobs()
    run(world, tmp_path)
    document, _ = read_output(tmp_path, "alpha")
    audio = document["meralion"]["audio"]
    assert audio["input"] == "/synthetic/alpha.wav" and audio["num_samples"] == synthetic.SAMPLES_120S and audio["seconds"] == 120.0
    assert audio["sha256"] == json.loads((tmp_path / "run" / "state" / "inv-1" / "recordings" / "alpha.json").read_text(encoding="utf-8"))["audio_sha256"]


def test_the_retry_block_records_the_decode_the_outcome_and_the_counts(tmp_path):
    run(two_jobs(), tmp_path)
    document, _ = read_output(tmp_path, "alpha")
    retries = document["meralion"]["retries"]
    assert retries["selected"] == 1 and retries["skipped"] == [] and (retries["inserted"], retries["adopted"], retries["failed"]) == (1, 0, 0)
    assert [d["retry_id"] for d in retries["decodes"]] == ["alpha_r000"] and retries["decodes"][0]["window"] == [600000, 1080000]
    assert [(d["kind"], d["result"]) for o in retries["outcomes"] for d in o["decisions"]] == [("insert", "inserted")]


def test_the_trace_carries_every_window_with_its_decode_result_seams_and_provenance(tmp_path):
    run(two_jobs(), tmp_path)
    document, _ = read_output(tmp_path, "alpha")
    trace = document["meralion"]["trace"]
    assert len(trace["windows"]) == 7 and [w["start_sample"] for w in trace["windows"]][:2] == [0, 240000]
    assert set(trace["windows"][0]) == {"start_sample", "end_sample", "text", "tokens", "stop_reason", "decode_seconds", "native"}
    assert trace["windows"][0]["tokens"] is None and trace["windows"][0]["stop_reason"] is None                 # the replay decoder has no telemetry: unavailable, not 0
    assert len(trace["chain"]) == 8 and len(trace["seams"]) == 7 and trace["provenance"]["policy"] == "m2ax_seq_ap_lm"          # 7 windows + 1 inserted retry, so 7 seams


def test_the_timing_block_is_present(tmp_path):
    run(two_jobs(), tmp_path)
    timing = read_output(tmp_path, "alpha")[0]["meralion"]["timing"]
    assert set(timing) == {"model_check_s", "model_load_s", "decode_s", "total_s"}


def test_the_document_is_written_as_utf8_without_ascii_escaping(tmp_path):
    world = synthetic.World({"zh": dict(break_at=2, replace_first="好的 谢谢您")})
    run(world, tmp_path)
    _, data = read_output(tmp_path, "zh")
    text = data.decode("utf-8")
    assert "好的 谢谢您" in text and "\\u597d" not in text


def test_a_failed_input_has_no_output_and_the_others_are_written(tmp_path):
    world = two_jobs()
    world.failing["beta"] = "Unsupported or unknown media format"
    summary = run(world, tmp_path)
    assert summary.exit_code == 2 and set(raw_files(tmp_path / "run" / "out")) == {"transcripts/alpha.json"}


def test_outputs_are_written_exclusively_and_never_overwritten(tmp_path):
    world = two_jobs()
    run(world, tmp_path)
    path = tmp_path / "run" / "out" / "transcripts" / "alpha.json"
    before = path.read_bytes()
    run(world, tmp_path)                                                                  # a second run of the same invocation
    assert path.read_bytes() == before


def test_the_default_effective_config_is_exactly_the_packaged_amt1_file():
    effective = synthetic.effective()
    assert effective.data == json.loads(cfg.DEFAULT_CONFIG.read_text(encoding="utf-8")) and effective.formatter == "percentages-v2+amount-v1"


# ---- the two configurations through the real pipeline (Stage 2A): the amount comma, and the legacy configuration preserved ----------------------------------------------------

AMOUNT = "two million, five hundred thousand dollars"


def amount_world():
    return synthetic.World({"amt": dict(break_at=2, replace_first=AMOUNT), "pct": dict(break_at=2, replace_first="half a percent")})


def run_with(effective, tmp_path, base):
    world = amount_world()
    summary = run(world, tmp_path, effective=effective, base=base)
    assert summary.exit_code == 0
    return {name: read_output(tmp_path, name, base=base)[0] for name in ("amt", "pct")}


def test_the_default_config_removes_the_amount_comma_and_rewrites_the_percentage(tmp_path):
    documents = run_with(synthetic.effective(), tmp_path, "amt1")
    amount, percent = documents["amt"], documents["pct"]
    assert AMOUNT in amount["meralion"]["transcript_raw"]                                           # the raw text keeps the comma
    assert "two million five hundred thousand dollars" in amount["results"]["transcripts"][0]["transcript"] and AMOUNT not in amount["results"]["transcripts"][0]["transcript"]
    assert amount["meralion"]["formatter"] == {"name": "percentages-v2+amount-v1", "version": "2+amount-v1", "changed": True}
    assert "0.5 percent" in percent["results"]["transcripts"][0]["transcript"] and "half a percent" in percent["meralion"]["transcript_raw"]


def test_the_legacy_config_is_preserved_and_leaves_the_amount_comma(tmp_path):
    documents = run_with(synthetic.legacy_effective(), tmp_path, "legacy")
    amount, percent = documents["amt"], documents["pct"]
    assert AMOUNT in amount["results"]["transcripts"][0]["transcript"] and amount["meralion"]["formatter"] == {"name": "percentages-v2", "version": "2", "changed": False}
    assert "0.5 percent" in percent["results"]["transcripts"][0]["transcript"] and percent["meralion"]["formatter"] == {"name": "percentages-v2", "version": "2", "changed": True}
    assert amount["meralion"]["config"]["config_id"] == "fg175-m2ax-seq-ap-lmonly-m5-fee2"


def test_the_raw_transcript_and_the_retry_behaviour_do_not_depend_on_the_formatter(tmp_path):
    new, old = run_with(synthetic.budgeted_effective(), tmp_path, "amt1"), run_with(synthetic.legacy_effective(), tmp_path, "legacy")      # same strategy, different formatter
    for name in ("amt", "pct"):
        assert new[name]["meralion"]["transcript_raw"] == old[name]["meralion"]["transcript_raw"]
        assert new[name]["meralion"]["retries"] == old[name]["meralion"]["retries"] and new[name]["meralion"]["trace"] == old[name]["meralion"]["trace"]
        assert new[name]["meralion"]["invocation"]["plan_sha256"] == old[name]["meralion"]["invocation"]["plan_sha256"]


def test_the_two_configs_have_different_fingerprints_in_their_outputs(tmp_path):
    new, old = run_with(synthetic.effective(), tmp_path, "amt1"), run_with(synthetic.legacy_effective(), tmp_path, "legacy")
    assert new["pct"]["meralion"]["config_fingerprint"] != old["pct"]["meralion"]["config_fingerprint"]
