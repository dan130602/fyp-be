"""Concurrent network only; actual verification lifecycle remains serialized and resumable."""
import threading
import pytest

from fyp_evaluation.llm.gemini_cohort import run_cohort
from fyp_evaluation.llm.errors import AmbiguousError,CapacityError
from fyp_evaluation.llm.fake_gemini_transport import call_id_of
from g5_support import make_runner,good_responder,report_body,mismatch_report

CALLS=['CALL_021','CALL_041','CALL_022']


def test_overlapping_network_sends_preserve_request_acceptance_and_reconstruction(tmp_path):
    runner,fake=make_runner(tmp_path)
    original=good_responder(runner)
    barrier=threading.Barrier(2)
    def respond(body):
        assert runner.store.read_attempt(call_id_of(body),'A001')['state']=='PREPARED'
        barrier.wait(timeout=10)
        return original(body)
    fake.responder=respond
    result=run_cohort(runner,CALLS[:2],max_workers=2)
    assert result['stopped'] is None
    assert len(fake.calls)==2 and all(a['state']=='ACCEPTED' for a in result['attempts'])
    assert runner.validate()==[] and runner.check()=={'missing':[],'changed':[],'extra':[]}
    assert all(runner.store.read_raw_response(c,'A001') for c in CALLS[:2])


def test_invalid_stops_new_requests_without_retrying_a_completed_judgement(tmp_path):
    runner,fake=make_runner(tmp_path)
    def respond(body):
        ctx=runner.builder.context(call_id_of(body))
        return report_body(mismatch_report(ctx,'C05','EV_001',ctx.planned_text('EV_001')[:30]))
    fake.responder=respond
    result=run_cohort(runner,CALLS,max_workers=1)
    assert result['stopped'] and len(fake.calls)==1
    assert runner.store.attempt_ids(CALLS[1])==[]
    assert result['attempts'][0]['state']=='ACCEPTED'


def test_ambiguous_request_is_preserved_possibly_billed_and_never_resent(tmp_path):
    runner,fake=make_runner(tmp_path,errors=[AmbiguousError('timeout')])
    result=run_cohort(runner,CALLS,max_workers=1)
    attempt=result['attempts'][0]
    assert len(fake.calls)==1 and attempt['possibly_billed'] and attempt['ambiguous']
    assert not runner.eligibility(CALLS[0])[0]
    assert runner.store.attempt_ids(CALLS[1])==[]


def test_capacity_retries_use_existing_policy_and_record_definite_unbilled_failure(tmp_path):
    runner,fake=make_runner(tmp_path,errors=[CapacityError('busy')]*4)
    result=run_cohort(runner,CALLS,max_workers=1)
    attempt=result['attempts'][0]
    assert len(fake.calls)==runner.flex.capacity_sends
    assert attempt['state']=='TRANSPORT_FAILED' and not attempt['possibly_billed']
    assert attempt['estimated_cost_usd']==0


def test_budget_callback_can_refuse_before_an_attempt_or_network_request(tmp_path):
    runner,fake=make_runner(tmp_path)
    def refuse(call,body):raise RuntimeError('budget ceiling')
    result=run_cohort(runner,CALLS,max_workers=2,before_prepare=refuse)
    assert result['stopped']=='budget ceiling' and fake.calls==[]
    assert all(runner.store.attempt_ids(c)==[] for c in CALLS)


def test_blocker_drains_existing_requests_but_does_not_launch_the_next_call(tmp_path):
    runner,fake=make_runner(tmp_path)
    barrier=threading.Barrier(2)
    good=good_responder(runner)
    def respond(body):
        barrier.wait(timeout=10)
        call=call_id_of(body)
        if call==CALLS[0]:
            ctx=runner.builder.context(call)
            return report_body(mismatch_report(ctx,'C05','EV_001',ctx.planned_text('EV_001')[:30]))
        return good(body)
    fake.responder=respond
    result=run_cohort(runner,CALLS,max_workers=2)
    assert result['stopped'] and len(fake.calls)==2
    assert all(a['state']=='ACCEPTED' for a in result['attempts'])
    assert runner.store.attempt_ids(CALLS[2])==[]


def test_stale_input_blocks_acceptance_and_further_sends(tmp_path):
    runner,fake=make_runner(tmp_path)
    good=good_responder(runner)
    original=runner._current_upstream_hashes
    changed=[False]
    runner._current_upstream_hashes=lambda call: {**original(call),**({'check_plan':'1'*64} if changed[0] else {})}
    def respond(body):
        result=good(body);changed[0]=True;return result
    fake.responder=respond
    result=run_cohort(runner,CALLS,max_workers=1)
    assert result['attempts'][0]['state']=='STALE_UPSTREAM' and len(fake.calls)==1


def test_stored_response_is_recovered_after_interrupted_adjudication_without_resend(tmp_path,monkeypatch):
    runner,fake=make_runner(tmp_path)
    adjudicate=runner._adjudicate
    def interrupted(attempt):raise RuntimeError('simulated interrupted adjudication')
    monkeypatch.setattr(runner,'_adjudicate',interrupted)
    result=run_cohort(runner,CALLS[:1],max_workers=1)
    assert result['attempts'][0]['state']=='RESPONDED'
    monkeypatch.setattr(runner,'_adjudicate',adjudicate)
    recovered=runner.run(CALLS[:1])
    assert recovered['recovered']==[(CALLS[0],'A001','ACCEPTED')]
    assert len(fake.calls)==1


def test_explicit_reopen_preserves_the_original_report_history(tmp_path):
    import json
    runner,fake=make_runner(tmp_path)
    assert run_cohort(runner,CALLS[:1],max_workers=1)['attempts'][0]['attempt_id']=='A001'
    result=run_cohort(runner,CALLS[:1],max_workers=1,reopen_reason='Artificial changed-input repair')
    assert result['attempts'][0]['attempt_id']=='A002'
    prov=json.loads((runner.root/runner.spec.provenance_path(CALLS[0])).read_text(encoding='utf-8'))
    assert [v['attempt_id'] for v in prov['accepted_history']]==['A001']
    assert len(fake.calls)==2 and runner.validate()==[]
