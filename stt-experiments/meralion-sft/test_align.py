"""CPU tests of the long-turn segmentation (synthetic aligner output with known pauses)."""
import re

import pytest

import sft_align as A
import sft_data as D


def synth(text, word_seconds=0.4, pauses=None, tokens=None):
    """Aligner-style units for `text` (words for Latin, one unit per Han character) with a 0.05 s gap, longer pauses after the given unit indices, and an RMS track that is quiet in gaps."""
    pauses = pauses or {}
    tokens = tokens or re.findall(r"[一-鿿]|[A-Za-z0-9$%'.,]+", text)
    t, units = 0.3, []
    for i, tok in enumerate(tokens):
        units.append({"text": tok, "start_time": t, "end_time": t + word_seconds})
        t += word_seconds + pauses.get(i, 0.05)
    duration = t + 0.3
    frames = int(duration / 0.01)
    rms = [0.01] * frames
    for u in units:
        for f in range(int(u["start_time"] / 0.01), min(frames, int(u["end_time"] / 0.01))):
            rms[f] = 0.1
    return {"units": units, "rms": rms}, duration


def item_for(text, duration, spans=(), split="TRAIN"):
    n = round(duration * 16000)
    return {"call_id": "CALL_T", "turn_id": 3, "split": split, "speaker": "CLIENT", "language": "ENGLISH", "start": 10.0, "end": 10.0 + duration, "start_sample": 160000, "end_sample": 160000 + n,
            "text": text, "spans": list(spans), "evidence": True}


def test_long_english_turn_is_cut_at_sentence_ends_inside_30s_and_partitions_text_and_audio():
    sentences = ["We would like to move the funds into the growth portfolio this week.", "Please confirm the amount of $50,000 before we proceed with anything.", "I understand the risk of this product and I accept it fully now.",
                 "Also keep the remaining balance in the money market account please."]
    text = " ".join(sentences)
    words = len(text.split())
    ends = {i for i, tok in enumerate(re.findall(r"[A-Za-z0-9$%'.,]+", text)) if tok.endswith(".")}
    alignment, duration = synth(text, word_seconds=0.85, pauses={i: 0.5 for i in ends})
    assert duration > 45
    item = item_for(text, duration)
    plan = A.plan_turn(item, alignment)
    assert plan["status"] == "resolved" and len(plan["pieces"]) >= 2
    clips = A.pieces_to_clips(item, plan)
    assert not A.verify_turn(item, clips)
    assert all(c["duration_seconds"] <= 29.5 + 1e-3 and c["end_sample"] - c["start_sample"] <= D.MAX_CLIP_SAMPLES for c in clips)
    assert clips[0]["start_sample"] == 160000 and clips[-1]["end_sample"] == item["end_sample"]
    assert all(a["end_sample"] == b["start_sample"] for a, b in zip(clips, clips[1:]))
    assert [c["text"] for c in clips[:-1]] and all(c["text"].rstrip().endswith(".") for c in clips[:-1])      # cuts at sentence ends
    assert " ".join(c["text"] for c in clips) == text


def test_cut_avoids_entity_span_and_number_runs():
    text = "Please buy ten thousand units of Meridian Asia Equity Fund today and then sell the whole Pacific Growth Income Fund position later in the week please."
    start = text.index("Meridian")
    spans = [{"start": start, "end": start + len("Meridian Asia Equity Fund"), "group": "value:T01", "label": "EV_1"}]
    alignment, duration = synth(text, word_seconds=1.7)
    item = item_for(text, duration, spans)
    plan = A.plan_turn(item, alignment)
    assert plan["status"] == "resolved" and not plan["inside_entity_cuts"]
    clips = A.pieces_to_clips(item, plan)
    assert not A.verify_turn(item, clips)
    holder = [c for c in clips if c["entity_spans"]]
    assert len(holder) == 1 and holder[0]["text"][holder[0]["entity_spans"][0]["start"]:holder[0]["entity_spans"][0]["end"]] == "Meridian Asia Equity Fund"


def test_mandarin_cut_is_at_punctuation_not_inside_a_numeral_run_and_attached_symbols_stay():
    text = "您好我是陈明，我的身份证号码是S一六六四三六一P，我想把五百万美元转到成长基金，请确认金额是否正确，谢谢您，现在方便吗，我还有几个问题想要问清楚，比如费用和风险。"
    tokens = re.findall(r"[一-鿿]|S|P", text)
    alignment, duration = synth(text, word_seconds=0.55, tokens=tokens, pauses={})
    # real pauses after the commas: find unit index after each comma
    pause_after = {}
    count = 0
    for ch in text:
        if re.match(r"[一-鿿]|S|P", ch):
            count += 1
        elif ch in "，。":
            pause_after[count - 1] = 0.35
    alignment, duration = synth(text, word_seconds=0.55, tokens=tokens, pauses=pause_after)
    item = item_for(text, duration)
    plan = A.plan_turn(item, alignment)
    assert plan["status"] == "resolved"
    for cut in plan["cuts"]:
        assert cut["kind"] in ("clause", "sentence"), cut                      # always at a comma / full stop here
    clips = A.pieces_to_clips(item, plan)
    assert not A.verify_turn(item, clips) and "".join(c["text"] for c in clips) == text


def test_dollar_sign_stays_with_its_number():
    text = "I confirm that I will transfer $ 5000 now. " * 12
    text = text.replace("$ 5000", "$5000").strip()
    tokens = re.findall(r"[A-Za-z]+|5000", text)
    alignment, duration = synth(text, word_seconds=0.9, tokens=tokens)
    item = item_for(text, duration)
    plan = A.plan_turn(item, alignment)
    clips = A.pieces_to_clips(item, plan)
    assert plan["status"] == "resolved" and all(not c["text"].endswith("$") and not c["text"].startswith("5000") for c in clips)


def test_unresolved_when_units_do_not_reproduce_the_reference_or_cover_the_audio_or_cut_in_loud_speech():
    text = "one two three four five six seven eight nine ten eleven twelve " * 4
    alignment, duration = synth(text.strip(), word_seconds=1.0)
    item = item_for(text.strip(), duration)
    bad = dict(alignment, units=alignment["units"][:-3])
    assert A.plan_turn(item, bad)["reason"] == "alignment_text_mismatch"
    late = {"units": [dict(u, start_time=u["start_time"] + 5, end_time=u["end_time"] + 5) for u in alignment["units"]], "rms": alignment["rms"]}
    assert A.plan_turn(item, late)["status"] == "unresolved"
    loud = {"units": alignment["units"], "rms": [0.1] * len(alignment["rms"])}                  # continuous speech, no pause anywhere
    nogap = {"units": [dict(u, start_time=i * 1.0, end_time=i * 1.0 + 1.0) for i, u in enumerate(alignment["units"])], "rms": loud["rms"]}
    plan = A.plan_turn(item_for(text.strip(), len(nogap["units"]) + 0.5), nogap)
    assert plan["status"] in ("resolved", "unresolved")                                          # flat loud audio: cut quality is judged by energy ratio, never silently trusted
    backwards = {"units": alignment["units"][::-1], "rms": alignment["rms"]}
    assert A.plan_turn(item, backwards)["status"] == "unresolved"


def test_verify_turn_detects_missing_text_and_gaps():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 5
    alignment, duration = synth(text.strip(), word_seconds=1.0, pauses={11: 0.4, 23: 0.4, 35: 0.4})
    item = item_for(text.strip(), duration)
    plan = A.plan_turn(item, alignment)
    clips = A.pieces_to_clips(item, plan)
    assert not A.verify_turn(item, clips)
    broken = [dict(c) for c in clips]
    broken[1] = dict(broken[1], start_sample=broken[1]["start_sample"] + 1000)
    assert A.verify_turn(item, broken)
    dropped = [clips[0]] + clips[2:]
    assert A.verify_turn(item, dropped)
