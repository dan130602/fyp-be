"""Artificial text fixtures for the SSML converter; no AWS requests."""

import json
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import convert_ssml
import export_ssml


def test_conversion_preserves_text_and_escapes_xml():
    text = "Welcome & thank you.\nThe amount is < 100."
    ssml = convert_ssml.convert_text(text)
    assert ElementTree.fromstring(ssml).text == text
    assert "&amp;" in ssml and "&lt;" in ssml


@pytest.mark.parametrize("text", ["", "  \n", "Invalid\x00text"])
def test_rejects_empty_or_invalid_xml_text(text):
    with pytest.raises(ValueError):
        convert_ssml.convert_text(text)


def test_file_conversion_protects_existing_files(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("Hello & welcome.", encoding="utf-8")
    output = source.with_suffix(".ssml")
    assert convert_ssml.main(["--input", str(source)]) == 0
    assert ElementTree.fromstring(output.read_text(encoding="utf-8")).text == "Hello & welcome."
    source.write_text("New text.", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        convert_ssml.main(["--input", str(source)])
    assert error.value.code == 1
    assert "Hello &amp; welcome." in output.read_text(encoding="utf-8")
    assert convert_ssml.main(["--input", str(source), "--overwrite"]) == 0
    assert "New text." in output.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        convert_ssml.main(["--input", str(source), "--output", str(source), "--overwrite"])
    assert source.read_text(encoding="utf-8") == "New text."


def test_call_conversion_keeps_turn_ids_and_speaker_roles(tmp_path, monkeypatch):
    turns = [
        {"turn_id": 0, "speaker": "EMPLOYEE", "text": "Welcome & hello."},
        {"turn_id": 1, "speaker": "CLIENT", "text": "Thank you."},
    ]
    dataset_root = tmp_path / "dataset"
    calls_dir = dataset_root / "calls"
    monkeypatch.setattr(export_ssml, "load_turns", lambda call_id, _calls_dir: turns)
    assert convert_ssml.main([
        "--dataset-root", str(dataset_root), "--call", "CALL_999"
    ]) == 0
    render = calls_dir / "CALL_999" / "renders" / "clean_voice_A"
    manifest = json.loads((render / "ssml_manifest.json").read_text(encoding="utf-8"))
    assert [(turn["turn_id"], turn["speaker"]) for turn in manifest["turns"]] == [
        (0, "EMPLOYEE"), (1, "CLIENT")
    ]
    assert (render / "segments" / "0.ssml").read_text(encoding="utf-8") == "<speak>Welcome &amp; hello.</speak>"
    assert (render / "segments" / "1.ssml").read_text(encoding="utf-8") == "<speak>Thank you.</speak>"
