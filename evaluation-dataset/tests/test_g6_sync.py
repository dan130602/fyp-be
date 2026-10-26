"""G6 synchronous generation (`--sync`): the same request, validation and acceptance as Batch, with no hidden SDK retry and an explicit
gate after an unknown outcome. Offline only; fake transports and a stub SDK client, nothing is sent anywhere."""

import json
from types import SimpleNamespace

import pytest

import g3_support
import multilingual_support as M
from fyp_evaluation.call_specs.compile import parse_json_text, serialize_json
from fyp_evaluation.dialogues import config
from fyp_evaluation.dialogues.cli import main
from fyp_evaluation.dialogues.feasibility import synthesize_dialogue
from fyp_evaluation.dialogues.run import DialogueRunner, RunError
from fyp_evaluation.llm.errors import AmbiguousError, FatalRequestError, RetryableError
from fyp_evaluation.llm.fake_transport import response_body
from fyp_evaluation.llm.openai_transport import OpenAITransport
from g6_support import SCHEMAS, Clock, make_runner, make_world, shared_inputs, submit_and_finish

X, Y = "CALL_027", "CALL_010"


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def artifact_text(runner, call_id=X):
    return (runner.root / runner.spec.artifact_path(call_id)).read_text(encoding="utf-8")


def provenance(runner, call_id=X):
    return parse_json_text((runner.root / runner.spec.provenance_path(call_id)).read_text(encoding="utf-8"))


# ============================================================================================ equivalence with Batch
def test_batch_and_sync_send_the_same_request_and_accept_the_same_dialogue(tmp_path):
    batch, batch_fake = make_runner(tmp_path / "batch")
    submit_and_finish(batch, batch_fake, [X])
    batch.collect()
    sync, sync_fake = make_runner(tmp_path / "sync")
    attempt = sync.run_sync(X)
    assert attempt["state"] == "ACCEPTED" and attempt["transport"] == "sync" and sync_fake.calls == [("create_response", X)]
    # the model-facing request is byte-identical: model, messages, response schema, reasoning settings, token cap, metadata
    assert batch.store.read_request(X, "A001") == sync.store.read_request(X, "A001")
    a, b = batch.store.read_attempt(X, "A001"), sync.store.read_attempt(X, "A001")
    for key in ("request_sha256", "prompt_template_sha256", "response_schema_sha256", "params", "upstream_sha256", "model", "price_table_version"):
        assert a[key] == b[key], key
    assert (a["transport"], b["transport"]) == ("batch", "sync")
    # the accepted artifact, the record and the validation outcome do not depend on the transport
    assert artifact_text(batch) == artifact_text(sync)
    assert batch.validate() == [] and sync.validate() == []
    assert (batch.root / config.RECORD).read_text(encoding="utf-8") == (sync.root / config.RECORD).read_text(encoding="utf-8")
    assert sync.check() == {"missing": [], "changed": [], "extra": []}
    assert provenance(sync)["attempts"][0]["transport"] == "sync" and provenance(batch)["attempts"][0]["transport"] == "batch"


def test_a_synchronous_attempt_is_priced_at_the_standard_rate_and_the_dry_run_shows_both_estimates(tmp_path):
    runner, _ = make_runner(tmp_path)
    attempt = runner.run_sync(X)
    batch_runner, batch_fake = make_runner(tmp_path / "b")
    submit_and_finish(batch_runner, batch_fake, [X])
    batch_runner.collect()
    assert attempt["estimated_cost_usd"] == pytest.approx(2 * batch_runner.store.read_attempt(X, "A001")["estimated_cost_usd"])
    estimate = make_runner(tmp_path / "d")[0].dry_run([X])["estimate"]
    assert estimate["sync_cost_usd"] == pytest.approx(2 * estimate["batch_cost_usd"])


def bad_response(ctx):
    response = synthesize_dialogue(ctx)
    response["turns"][0]["text"] = "No markers here."
    return response


def test_a_structurally_invalid_dialogue_is_rejected_and_never_accepted(tmp_path):
    runner, fake = make_runner(tmp_path, overrides={X: bad_response})
    attempt = runner.run_sync(X)
    assert attempt["state"] == "REJECTED" and attempt["validation_errors"] and not (runner.root / runner.spec.artifact_path(X)).exists()
    with pytest.raises(RunError, match="REJECTED|rejected"):  # retried only on request, exactly like Batch
        runner.run_sync(X)
    assert len(fake.calls) == 1


def test_a_provider_status_failure_is_recorded_without_acceptance(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.responder = lambda body: response_body("", status="incomplete", incomplete_reason="max_output_tokens")
    attempt = runner.run_sync(X)
    assert attempt["state"] in ("REJECTED", "TRANSPORT_FAILED") and not (runner.root / runner.spec.artifact_path(X)).exists()


# ============================================================================================ upstream freshness and eligibility
def test_a_changed_upstream_closes_a_synchronous_attempt_as_stale(tmp_path):
    runner, fake = make_world(tmp_path)
    good = fake.responder

    def change_upstream_then_answer(body):
        path = runner.root / "authoring/plan_verifications" / f"{X}.json"
        data = parse_json_text(path.read_bytes().decode("utf-8"))
        data["rationale_touched"] = True
        path.write_bytes(serialize_json(data).encode("utf-8"))
        return good(body)

    fake.responder = change_upstream_then_answer
    attempt = runner.run_sync(X)
    assert attempt["state"] == "STALE_UPSTREAM" and not (runner.root / runner.spec.artifact_path(X)).exists()


def test_a_call_without_a_valid_g5_report_is_not_sent(tmp_path):
    runner, fake = make_world(tmp_path)
    (runner.root / "authoring/plan_verifications" / f"{X}.json").unlink()
    with pytest.raises(RunError, match="G5"):
        runner.run_sync(X)
    assert fake.calls == []


# ============================================================================================ unknown outcomes
def test_an_unknown_outcome_is_possibly_billed_blocks_the_call_and_is_never_resent(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.sync_error = AmbiguousError("APITimeoutError: Request timed out.")
    attempt = runner.run_sync(X)
    assert attempt["state"] == "TRANSPORT_FAILED" and attempt["possibly_billed"] is True and attempt["ambiguous"] is True
    assert provenance(runner)["attempts"][0]["possibly_billed"] is True
    with pytest.raises(RunError, match="AWAITING_ACKNOWLEDGEMENT"):
        runner.run_sync(X)
    assert len(fake.calls) == 1, "nothing was re-sent"
    again = runner.run_sync(X, acknowledge_possible_duplicate=True)
    assert again["state"] == "ACCEPTED" and again["attempt_id"] == "A002" and len(fake.calls) == 2


def test_a_definite_rate_limit_is_recorded_and_not_ambiguous(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.sync_error = RetryableError("RateLimitError: slow down")
    attempt = runner.run_sync(X)
    assert attempt["state"] == "TRANSPORT_FAILED" and not attempt.get("ambiguous")
    assert runner.run_sync(X)["state"] == "ACCEPTED"  # a definite capacity rejection needs no acknowledgement


def test_a_fatal_error_stops_the_run(tmp_path):
    runner, fake = make_runner(tmp_path)
    fake.sync_error = FatalRequestError("BadRequestError: schema rejected")
    with pytest.raises(FatalRequestError):
        runner.run_sync(X)


def test_acknowledging_without_an_ambiguous_attempt_is_an_error(tmp_path):
    runner, _ = make_runner(tmp_path)
    with pytest.raises(RunError, match="no unacknowledged ambiguous attempt"):
        runner.run_sync(X, acknowledge_possible_duplicate=True)


# ============================================================================================ the transport: no hidden retries
class StubClient:
    """Records which SDK options each call was made under; `with_options` returns a derived client like the real SDK."""

    def __init__(self, log, options=None, fail=None):
        self.log, self.options, self.fail = log, dict(options or {}), fail
        self.responses = SimpleNamespace(create=self._create)

    def with_options(self, **options):
        return StubClient(self.log, {**self.options, **options}, self.fail)

    def _create(self, **body):
        self.log.append(dict(self.options))
        if self.fail:
            raise self.fail
        return SimpleNamespace(model_dump=lambda mode: {"id": "resp", "status": "completed"})


def _errors():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "http://invalid")
    return SimpleNamespace(
        timeout=openai.APITimeoutError(request=request), connection=openai.APIConnectionError(request=request),
        server=openai.InternalServerError("boom", response=httpx.Response(500, request=request), body=None),
        rate=openai.RateLimitError("slow", response=httpx.Response(429, request=request), body=None),
        bad=openai.BadRequestError("bad", response=httpx.Response(400, request=request), body=None))


def test_the_g6_transport_options_turn_sdk_retries_off_and_report_unknown_outcomes_as_ambiguous():
    errors = _errors()
    log = []
    g6 = OpenAITransport(StubClient(log), response_max_retries=0, response_timeout=config.SYNC_TIMEOUT_SECONDS, ambiguous_sync=True)
    assert g6.create_response({"model": "m"}) == {"id": "resp", "status": "completed"}
    assert log == [{"max_retries": 0, "timeout": config.SYNC_TIMEOUT_SECONDS}]
    for name in ("timeout", "connection", "server"):
        with pytest.raises(AmbiguousError):
            OpenAITransport(StubClient([], fail=getattr(errors, name)), response_max_retries=0, ambiguous_sync=True).create_response({})
    with pytest.raises(RetryableError) as rate:  # a definite rate limit was not processed: not ambiguous
        OpenAITransport(StubClient([], fail=errors.rate), response_max_retries=0, ambiguous_sync=True).create_response({})
    assert not isinstance(rate.value, AmbiguousError)
    with pytest.raises(FatalRequestError):
        OpenAITransport(StubClient([], fail=errors.bad), response_max_retries=0, ambiguous_sync=True).create_response({})


def test_every_other_stage_keeps_the_shared_transport_behaviour():
    errors = _errors()
    log = []
    default = OpenAITransport(StubClient(log))
    default.create_response({})
    assert log == [{}], "no per-request override: the client's own SDK retries apply, exactly as before"
    with pytest.raises(RetryableError) as raised:
        OpenAITransport(StubClient([], fail=errors.timeout)).create_response({})
    assert not isinstance(raised.value, AmbiguousError)


def test_the_real_transport_factories_are_chosen_per_stage(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(OpenAITransport, "from_env", classmethod(lambda cls, project_root=None, **options: captured.update(options) or "transport"))
    g6, _ = make_runner(tmp_path)
    assert g6._default_transport_factory()() == "transport"
    assert captured == {"response_max_retries": 0, "response_timeout": config.SYNC_TIMEOUT_SECONDS, "ambiguous_sync": True}
    captured.clear()
    (tmp_path / "g3").mkdir()
    g3, _ = g3_support.make_runner(tmp_path / "g3")
    assert g3._default_transport_factory()() == "transport" and captured == {}


# ============================================================================================ the command line
def cli(runner, fake, *extra):
    argv = ["--dataset-root", str(runner.root), "--audit-dir", str(runner.audit_root), "--schema-dir", str(SCHEMAS), *extra]
    return main(argv, transport=fake, clock=Clock(),
                runner_class=lambda root, **kwargs: DialogueRunner(root, inputs=shared_inputs(), **kwargs))


def test_the_command_line_runs_one_call_synchronously(tmp_path, capsys):
    runner, fake = make_runner(tmp_path)
    assert cli(runner, fake, "--sync", "--call-id", X) == 0 and f"{X} A001 ACCEPTED" in capsys.readouterr().out
    with pytest.raises(SystemExit):  # exactly one call per synchronous invocation
        cli(runner, fake, "--sync", "--call-id", X, "--call-id", Y)
    assert [c for c in fake.calls] == [("create_response", X)]


def test_acknowledgement_is_offered_only_to_stages_that_can_have_an_unknown_outcome(tmp_path, capsys):
    runner, fake = make_runner(tmp_path)
    fake.sync_error = AmbiguousError("timeout")
    assert cli(runner, fake, "--sync", "--call-id", X) == 1
    capsys.readouterr()
    assert cli(runner, fake, "--sync", "--call-id", X) == 2 and "AWAITING_ACKNOWLEDGEMENT" in capsys.readouterr().err
    assert cli(runner, fake, "--sync", "--call-id", X, "--acknowledge-duplicate-risk") == 0
    with pytest.raises(SystemExit):
        cli(runner, fake, "--submit", "--acknowledge-duplicate-risk")  # still not valid with --submit
    (tmp_path / "g3").mkdir()
    g3_runner, _ = g3_support.make_runner(tmp_path / "g3")
    from fyp_evaluation.call_configs.cli import main as g3_main
    argv = ["--dataset-root", str(g3_runner.root), "--sync", "--call-id", "CALL_001", "--acknowledge-duplicate-risk"]
    assert g3_main(argv, runner_class=lambda root, **kwargs: g3_runner) == 2 and "not supported by this stage" in capsys.readouterr().err


# ============================================================================================ selected calls on the multilingual release
def test_a_selected_multilingual_call_is_generated_synchronously_under_its_own_policy(tmp_path):
    singlish, mandarin = M.first_call("SINGLISH"), M.first_call("MANDARIN")
    chain = M.chain(tmp_path, [singlish, mandarin])
    assert set(chain.run_g3().values()) == set(chain.run_g4().values()) == set(chain.run_g5().values()) == {"ACCEPTED"}
    runner = chain.open_g6()
    attempt = runner.run_sync(mandarin)
    assert attempt["state"] == "ACCEPTED" and attempt["transport"] == "sync"
    dialogue = json.loads((chain.root / "authoring/dialogues" / f"{mandarin}.json").read_text(encoding="utf-8"))
    assert dialogue["schema_version"] == "2.0" and dialogue["length"]["unit"] == "han_chars"
    assert runner.run_sync(singlish)["state"] == "ACCEPTED"
    guide = json.loads((chain.root / "authoring/dialogues" / f"{singlish}.json").read_text(encoding="utf-8"))["source"]
    assert "register_guide_sha256" in guide
    other = next(c for c in M.upstream_inputs().call_ids if c not in (singlish, mandarin))
    with pytest.raises(RunError):
        runner.run_sync(other)  # no accepted G3 config: not eligible, nothing sent
    assert runner.validate() == []
