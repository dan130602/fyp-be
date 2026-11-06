"""G8 AWS credential loading: dummy values only, no AWS request, no real .env is read."""

import pytest

from fyp_evaluation.llm.errors import CredentialsError
from fyp_evaluation.rendering.credentials import aws_credentials
from fyp_evaluation.rendering.cli import main

DUMMY_ID, DUMMY_SECRET, DUMMY_TOKEN = "AKIAdummyDUMMYdummy", "dummySECRETdummySECRETdummySECRET00", "dummyTOKENdummy"


def write_env(root, text):
    (root / ".env").write_text(text, encoding="utf-8")


def test_the_env_file_supplies_the_pair_and_the_session_token_is_optional(tmp_path):
    write_env(tmp_path, f"AWS_ACCESS_KEY_ID={DUMMY_ID}\nAWS_SECRET_ACCESS_KEY={DUMMY_SECRET}\nAWS_DEFAULT_REGION=us-east-1\nOPENAI_API_KEY=sk-unrelated\n")
    assert aws_credentials(tmp_path, environ={}) == {"aws_access_key_id": DUMMY_ID, "aws_secret_access_key": DUMMY_SECRET}
    write_env(tmp_path, f"AWS_ACCESS_KEY_ID={DUMMY_ID}\nAWS_SECRET_ACCESS_KEY={DUMMY_SECRET}\nAWS_SESSION_TOKEN={DUMMY_TOKEN}\n")
    assert aws_credentials(tmp_path, environ={})["aws_session_token"] == DUMMY_TOKEN


def test_existing_environment_variables_win_and_are_never_mixed_with_the_file(tmp_path):
    write_env(tmp_path, f"AWS_ACCESS_KEY_ID=file{DUMMY_ID}\nAWS_SECRET_ACCESS_KEY=file{DUMMY_SECRET}\nAWS_SESSION_TOKEN=file{DUMMY_TOKEN}\n")
    environment = {"AWS_ACCESS_KEY_ID": DUMMY_ID, "AWS_SECRET_ACCESS_KEY": DUMMY_SECRET}
    assert aws_credentials(tmp_path, environ=environment) == {"aws_access_key_id": DUMMY_ID, "aws_secret_access_key": DUMMY_SECRET}  # no file token
    half = {"AWS_ACCESS_KEY_ID": DUMMY_ID}  # an incomplete environment pair is not completed from the file's other half
    assert aws_credentials(tmp_path, environ=half)["aws_access_key_id"] == f"file{DUMMY_ID}"
    assert half == {"AWS_ACCESS_KEY_ID": DUMMY_ID}  # the environment is never mutated


def test_a_missing_or_incomplete_pair_is_refused_naming_variables_not_values(tmp_path):
    with pytest.raises(CredentialsError, match="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY") as stopped:
        aws_credentials(tmp_path, environ={})
    write_env(tmp_path, f"AWS_ACCESS_KEY_ID={DUMMY_ID}\nAWS_SECRET_ACCESS_KEY=\n")
    with pytest.raises(CredentialsError) as incomplete:
        aws_credentials(tmp_path, environ={"AWS_SECRET_ACCESS_KEY": "   "})
    for error in (stopped.value, incomplete.value):
        assert DUMMY_ID not in str(error) and DUMMY_SECRET not in str(error) and "No request was made" in str(error)


def test_the_region_is_explicit_and_not_taken_from_the_environment(tmp_path, monkeypatch):
    from fyp_evaluation.rendering.cli import build_parser
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    assert build_parser().parse_args(["--dataset-root", "x"]).region == "ap-southeast-1"
    assert not any(action.dest == "profile" for action in build_parser()._actions)  # no --profile in this setup


def test_execute_without_credentials_stops_before_any_polly_request(monkeypatch, capsys, tmp_path):
    import fyp_evaluation.rendering.credentials as credentials
    monkeypatch.setattr(credentials, "PROJECT_ROOT", tmp_path)
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    from pathlib import Path
    release = Path(__file__).resolve().parents[1] / "datasets" / "advisory_execution_pilot_v1"
    assert main(["--dataset-root", str(release), "--execute", "--call-id", "CALL_001", "--render-id", "no_credentials_probe"]) == 2
    err = capsys.readouterr().err
    assert "are not both set" in err and "No request was made" in err
    assert not (release / "calls" / "CALL_001" / "renders" / "no_credentials_probe").exists()
