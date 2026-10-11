"""Artificial audio fixtures for the Polly renderer; no AWS requests."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path



SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_polly.py"
spec = importlib.util.spec_from_file_location("render_polly", SCRIPT)
render_polly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_polly)


class FakePolly:
    def __init__(self):
        self.requests = []

    def synthesize_speech(self, **request):
        self.requests.append(request)
        samples = 16000 if len(self.requests) == 1 else 8000
        return {"AudioStream": io.BytesIO(b"\x01\x00" * samples)}


def test_render_stitches_turns_and_preserves_canonical_data(tmp_path, monkeypatch):
    call_dir = tmp_path / "CALL_999"
    call_dir.mkdir()
    canonical = {"call_id": "CALL_999", "dialogue": [
        {"turn_id": 0, "speaker": "EMPLOYEE", "text": "Hello & welcome."},
        {"turn_id": 1, "speaker": "CLIENT", "text": "Thank you."},
    ]}
    source = call_dir / "ground_truth.json"
    source.write_text(json.dumps(canonical), encoding="utf-8")
    monkeypatch.setattr(render_polly, "encode_mp3",
                        lambda _ffmpeg, pcm, path: path.write_bytes(pcm[:4]))
    args = argparse.Namespace(render_id="test_render", overwrite=False, silence_ms=400,
                              employee_voice="Jasmine", client_voice="Matthew", engine="neural")
    polly = FakePolly()

    result = render_polly.render_call(
        "CALL_999",
        render_polly.load_turns("CALL_999", tmp_path),
        args,
        polly,
        "unused",
        tmp_path,
    )

    assert result.is_file()
    timings = json.loads(result.with_name("timings.json").read_text(encoding="utf-8"))
    assert timings["duration_seconds"] == 1.9
    assert timings["turns"] == [
        {"turn_id": 0, "start_seconds": 0, "end_seconds": 1},
        {"turn_id": 1, "start_seconds": 1.4, "end_seconds": 1.9},
    ]
    assert sorted(p.name for p in (result.parent / "segments").glob("*.mp3")) == ["0.mp3", "1.mp3"]
    assert [request["VoiceId"] for request in polly.requests] == ["Jasmine", "Matthew"]
    assert "Hello &amp; welcome." in polly.requests[0]["Text"]
    assert json.loads(source.read_text(encoding="utf-8")) == canonical
