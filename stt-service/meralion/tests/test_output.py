import copy
import json

import pytest

from meralion_worker import config as cfg
from meralion_worker import output as out
from meralion_worker.formatters import apply_formatter, formatter_version


def effective_for(formatter):
    data = json.loads(cfg.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    data["formatter"] = formatter
    return cfg.load_runtime_config(data, env={}).as_record()


def make(raw="the fee is half a percent", formatter="percentages-v2", job="job-1.A_b", **overrides):
    effective = effective_for(formatter)
    arguments = dict(
        job_name=job, transcript_raw=raw, transcript_final=apply_formatter(formatter, raw), effective=effective,
        invocation={"id": "inv-1", "members": 1}, decoder={"backend": "transformers", "telemetry": {"tokens": True}}, model={"manifest": "m", "revision": "r"},
        worker={"version": "0.1.0"}, audio={"input": "a.wav", "sha256": "0" * 64}, retries={"selected": 0}, trace={"windows": []}, timing={"total_s": 0.0})
    arguments.update(overrides)
    return out.build_document(**arguments)


def test_shape_and_required_fields():
    document = make()
    assert document["jobName"] == "job-1.A_b" and document["status"] == "COMPLETED"
    assert document["results"] == {"transcripts": [{"transcript": "the fee is 0.5 percent"}]}
    block = document["meralion"]
    assert block["schema_version"] == "meralion-worker-output-v0"
    assert block["config_fingerprint"] == effective_for("percentages-v2")["config_fingerprint"] and block["config"]["formatter"] == "percentages-v2"
    for key in ("config", "invocation", "decoder", "model", "worker", "audio", "retries", "trace", "timing"):
        assert key in block


def test_raw_and_final_are_both_kept_and_distinguishable():
    block = make()["meralion"]
    assert block["transcript_raw"] == "the fee is half a percent" and block["transcript_final"] == "the fee is 0.5 percent"
    assert block["formatter"] == {"name": "percentages-v2", "version": "2", "changed": True}


def test_formatter_none_leaves_the_transcript_unchanged():
    document = make(formatter="none")
    block = document["meralion"]
    assert block["transcript_raw"] == block["transcript_final"] == document["results"]["transcripts"][0]["transcript"]
    assert block["formatter"] == {"name": "none", "version": None, "changed": False}


def test_a_none_formatter_that_changed_the_text_is_rejected():
    document = make(formatter="none")
    document["meralion"]["transcript_final"] = "changed"
    document["results"]["transcripts"][0]["transcript"] = "changed"
    with pytest.raises(out.OutputError, match="must equal what formatter"):
        out.validate_document(document)


def test_results_holds_only_transcripts_and_no_invented_fields_anywhere_in_it():
    document = make()
    assert set(document["results"]) == {"transcripts"}
    encoded = json.dumps(document["results"])
    for forbidden in ("items", "speaker_label", "speaker_labels", "audio_segments", "channel_labels", "start_time", "end_time", "confidence", "alternatives"):
        assert forbidden not in encoded


@pytest.mark.parametrize("extra", [{"items": []}, {"speaker_labels": {}}, {"audio_segments": []}, {"channel_labels": {}}])
def test_extra_results_keys_are_rejected(extra):
    document = make()
    document["results"].update(extra)
    with pytest.raises(out.OutputError):
        out.validate_document(document)


def test_forbidden_keys_nested_inside_transcripts_are_rejected():
    document = make()
    document["results"]["transcripts"][0]["confidence"] = 0.9
    with pytest.raises(out.OutputError):
        out.validate_document(document)


def test_the_meralion_block_may_describe_the_decoder_without_leaking_into_results():
    document = make(decoder={"telemetry": {"tokens": None, "stop_reason": None}})        # unavailable telemetry stays None (never 0)
    assert document["meralion"]["decoder"]["telemetry"]["tokens"] is None
    assert "telemetry" not in json.dumps(document["results"])


@pytest.mark.parametrize("name", ["job-1", "A.b_c-9", "x" * 200])
def test_valid_job_names(name):
    assert make(job=name)["jobName"] == name


@pytest.mark.parametrize("name", ["", "has space", "slash/name", "über", "x" * 201, "a:b", "job-1\n", "\njob-1", "job-1\r\n", "job-1\n\n"])
def test_invalid_job_names_are_rejected(name):
    with pytest.raises(out.OutputError, match="jobName"):
        make(job=name)


def test_the_two_transcript_fields_must_agree():
    document = make()
    document["results"]["transcripts"][0]["transcript"] = "something else"
    with pytest.raises(out.OutputError, match="must equal"):
        out.validate_document(document)


def test_changed_flag_must_match_the_texts():
    document = make()
    document["meralion"]["formatter"]["changed"] = False
    with pytest.raises(out.OutputError, match="changed"):
        out.validate_document(document)


def test_serialisation_is_utf8_without_ascii_escaping_and_round_trips():
    raw = "费用是 half a percent，谢谢"
    document = make(raw=raw)
    data = out.serialize(document)
    text = data.decode("utf-8")
    assert "费用是" in text and "\\u" not in text and text.endswith("\n")
    assert json.loads(text) == document
    assert json.loads(text)["meralion"]["transcript_raw"] == raw                         # Chinese untouched by formatter v2


def test_build_does_not_mutate_its_inputs():
    effective = cfg.load_runtime_config(env={}).as_record()
    before = copy.deepcopy(effective)
    make(effective=effective)
    assert effective == before


# ---- formatter provenance must agree with the recorded effective configuration -----------------------------------------------------------------------------------------

def test_formatter_identity_is_derived_from_the_effective_config():
    assert make(formatter="percentages-v2")["meralion"]["formatter"] == {"name": "percentages-v2", "version": "2", "changed": True}
    assert make(formatter="none")["meralion"]["formatter"] == {"name": "none", "version": None, "changed": False}


def test_build_document_no_longer_takes_a_separate_formatter_name():
    with pytest.raises(TypeError):
        make(formatter_name="none")


def test_a_declared_formatter_that_contradicts_the_recorded_config_is_rejected():
    """Reproduces the reported case: the config says percentages-v2, the document declares 'none' and carries the unformatted text."""
    raw = "the fee is half a percent"
    document = make(raw=raw, formatter="percentages-v2")
    document["meralion"]["formatter"] = {"name": "none", "version": None, "changed": False}
    document["meralion"]["transcript_final"] = raw
    document["results"]["transcripts"][0]["transcript"] = raw
    with pytest.raises(out.OutputError, match="formatter in meralion.config"):
        out.validate_document(document)


def test_a_configured_formatter_that_was_not_applied_is_rejected():
    raw = "the fee is half a percent"
    document = make(raw=raw, formatter="percentages-v2")
    document["meralion"]["transcript_final"] = raw                        # still says percentages-v2 but the text was never formatted
    document["results"]["transcripts"][0]["transcript"] = raw
    document["meralion"]["formatter"]["changed"] = False
    with pytest.raises(out.OutputError, match="must equal what formatter"):
        out.validate_document(document)


def test_a_wrong_formatter_version_is_rejected():
    document = make()
    document["meralion"]["formatter"]["version"] = "1"
    with pytest.raises(out.OutputError, match="version"):
        out.validate_document(document)


def test_an_unknown_configured_formatter_is_rejected():
    document = make()
    document["meralion"]["config"]["formatter"] = "percentages-v9"
    with pytest.raises(out.OutputError, match="unknown formatter"):
        out.validate_document(document)


def test_the_recorded_fingerprint_must_match_the_recorded_config():
    document = make()
    document["meralion"]["config"]["config_id"] = "tampered"
    with pytest.raises(out.OutputError, match="fingerprint"):
        out.validate_document(document)
    document = make()
    document["meralion"]["config_fingerprint"] = "0" * 64
    with pytest.raises(out.OutputError, match="fingerprint"):
        out.validate_document(document)


# ---- the combined formatter's identity is validated like the others (Stage 2A) ---------------------------------------------------------------------------------------------------

COMBINED = "percentages-v2+amount-v1"
BOTH_RAW = "the fee is half a percent and the amount is two million, five hundred thousand dollars"


def test_the_combined_formatter_name_and_version_are_derived_from_the_config():
    block = make(raw=BOTH_RAW, formatter=COMBINED)["meralion"]
    assert block["formatter"] == {"name": COMBINED, "version": "2+amount-v1", "changed": True}
    assert block["transcript_raw"] == BOTH_RAW
    assert block["transcript_final"] == "the fee is 0.5 percent and the amount is two million five hundred thousand dollars"


def test_a_document_that_only_applied_the_percentage_step_is_rejected_under_the_combined_config():
    document = make(raw=BOTH_RAW, formatter=COMBINED)
    only_percentages = apply_formatter("percentages-v2", BOTH_RAW)
    document["meralion"]["transcript_final"] = only_percentages
    document["results"]["transcripts"][0]["transcript"] = only_percentages
    with pytest.raises(out.OutputError, match="must equal what formatter"):
        out.validate_document(document)


def test_a_document_declaring_the_legacy_formatter_under_the_combined_config_is_rejected():
    document = make(raw=BOTH_RAW, formatter=COMBINED)
    document["meralion"]["formatter"]["name"] = "percentages-v2"
    document["meralion"]["formatter"]["version"] = "2"
    with pytest.raises(out.OutputError, match="formatter in meralion.config"):
        out.validate_document(document)


def test_a_wrong_version_for_the_combined_formatter_is_rejected():
    for wrong in ("2", "amount-v1", "2+amount-v2", None):
        document = make(raw=BOTH_RAW, formatter=COMBINED)
        document["meralion"]["formatter"]["version"] = wrong
        with pytest.raises(out.OutputError, match="version"):
            out.validate_document(document)


def test_the_legacy_formatter_leaves_the_amount_comma_in_the_final_text():
    block = make(raw=BOTH_RAW, formatter="percentages-v2")["meralion"]
    assert block["transcript_final"] == "the fee is 0.5 percent and the amount is two million, five hundred thousand dollars"
    assert block["formatter"] == {"name": "percentages-v2", "version": "2", "changed": True}
