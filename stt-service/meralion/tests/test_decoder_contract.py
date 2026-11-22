import numpy as np
import pytest

from meralion_worker.decoders.base import MAX_WINDOW_SAMPLES, DecodeRequest, DecodeResult, Decoder, validate_piece
from meralion_worker.decoders.replay_decoder import ReplayDecoder, ReplayMissing, build_table


def piece(n=16000):
    return np.zeros(n, dtype=np.float32)


def test_a_result_round_trips_and_none_stays_none():
    result = DecodeResult(text="hello", tokens=None, stop_reason=None, decode_seconds=0.0, native={"finish_reason": None})
    again = DecodeResult.from_dict(result.to_dict())
    assert again == result and again.tokens is None and again.stop_reason is None and again.native == {"finish_reason": None}
    full = DecodeResult(text="x", tokens=0, stop_reason="end_token", decode_seconds=1.5, native={"batch_size": 3})
    assert DecodeResult.from_dict(full.to_dict()).tokens == 0                       # a real zero stays zero


@pytest.mark.parametrize("bad", [
    dict(text=5, tokens=1, stop_reason=None, decode_seconds=0.0),
    dict(text="x", tokens=-1, stop_reason=None, decode_seconds=0.0),
    dict(text="x", tokens=True, stop_reason=None, decode_seconds=0.0),
    dict(text="x", tokens=1.5, stop_reason=None, decode_seconds=0.0),
    dict(text="x", tokens=1, stop_reason="stop", decode_seconds=0.0),
    dict(text="x", tokens=1, stop_reason=None, decode_seconds=-1.0),
    dict(text="x", tokens=1, stop_reason=None, decode_seconds="1"),
    dict(text="x", tokens=1, stop_reason=None, decode_seconds=0.0, native=[]),
])
def test_invalid_results_are_rejected(bad):
    with pytest.raises(ValueError):
        DecodeResult(**bad)


def test_from_dict_requires_exactly_the_five_fields():
    with pytest.raises(ValueError):
        DecodeResult.from_dict({"text": "x"})
    with pytest.raises(ValueError):
        DecodeResult.from_dict({"text": "x", "tokens": 1, "stop_reason": None, "decode_seconds": 0.0, "native": {}, "extra": 1})


def test_piece_validation():
    validate_piece(piece(16000), 100, 16100)
    validate_piece(piece(MAX_WINDOW_SAMPLES), 0, MAX_WINDOW_SAMPLES)
    with pytest.raises(ValueError, match="1-D float32"):
        validate_piece(piece().astype(np.float64), 0, 16000)
    with pytest.raises(ValueError, match="1-D float32"):
        validate_piece(np.zeros((2, 8000), dtype=np.float32), 0, 16000)
    with pytest.raises(ValueError, match="1-D float32"):
        validate_piece([0.0] * 10, 0, 10)
    with pytest.raises(ValueError, match="samples"):
        validate_piece(piece(0), 0, 0)
    with pytest.raises(ValueError, match="samples"):
        validate_piece(piece(MAX_WINDOW_SAMPLES + 1), 0, MAX_WINDOW_SAMPLES + 1)
    with pytest.raises(ValueError, match="differs"):
        validate_piece(piece(16000), 0, 15999)


class Counting(Decoder):
    def __init__(self):
        self.calls = []

    def decode(self, piece, *, recording_id, start_sample, end_sample):
        self.calls.append((recording_id, start_sample, end_sample))
        return DecodeResult(text=f"{recording_id}:{start_sample}", tokens=1, stop_reason="end_token", decode_seconds=0.0)

    def describe(self):
        return {"backend": "counting", "settings_id": "s", "frozen_digest": "d", "runtime": {"packages": {}, "patches": []}}


def test_the_default_decode_many_loops_decode_in_order():
    decoder = Counting()
    requests = [DecodeRequest("a", 0, 16000, piece()), DecodeRequest("b", 16000, 32000, piece())]
    assert [r.text for r in decoder.decode_many(requests)] == ["a:0", "b:16000"]
    assert decoder.calls == [("a", 0, 16000), ("b", 16000, 32000)]
    assert decoder.identity() == {"backend": "counting", "settings_id": "s", "frozen_digest": "d", "runtime": {"packages": {}, "patches": []}}


def test_the_replay_decoder_looks_up_by_recording_and_bounds():
    decoder = ReplayDecoder({("rec", 0, 16000): "first", ("rec", 16000, 32000): "second"})
    assert decoder.decode(piece(), recording_id="rec", start_sample=0, end_sample=16000).text == "first"
    assert [r.text for r in decoder.decode_many([DecodeRequest("rec", 16000, 32000, piece())])] == ["second"]
    assert decoder.calls == [("rec", 0, 16000), ("rec", 16000, 32000)]
    assert decoder.describe()["backend"] == "replay"


def test_a_missing_replay_key_is_an_error_never_an_empty_text():
    decoder = ReplayDecoder({("rec", 0, 16000): "first"})
    with pytest.raises(ReplayMissing):
        decoder.decode(piece(), recording_id="rec", start_sample=16000, end_sample=32000)
    with pytest.raises(ReplayMissing):
        decoder.decode(piece(), recording_id="other", start_sample=0, end_sample=16000)


def test_a_table_with_one_key_and_two_texts_is_refused():
    with pytest.raises(ValueError, match="two different texts"):
        build_table([(("r", 0, 1), "a"), (("r", 0, 1), "b")])
    assert build_table([(("r", 0, 1), "a"), (("r", 0, 1), "a")])[("r", 0, 1)].text == "a"      # the same text twice is fine


def test_the_replay_decoder_validates_its_input_like_a_real_one():
    decoder = ReplayDecoder({("rec", 0, 16000): "x"})
    with pytest.raises(ValueError):
        decoder.decode(piece(8000), recording_id="rec", start_sample=0, end_sample=16000)
