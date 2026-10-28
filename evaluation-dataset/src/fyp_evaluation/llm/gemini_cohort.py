"""Bounded concurrent Gemini Flex sends with the existing verification lifecycle.

One stage lock is retained throughout; only network sends and raw-response writes
overlap. The parent alone closes attempts, adjudicates and updates provenance.
No provider Batch, model/tier fallback, altered payload or hidden retry is added.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .errors import AmbiguousError, CapacityError, TransportError
from .stage_runner import RunError


def run_cohort(runner, call_ids, *, max_workers=4, before_prepare=None, on_prepared=None, on_outcome=None, reopen_reason=None):
    if not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be between1 and8")
    report = {"attempts":[], "recovered":[], "skipped":{}, "stopped":None}
    with runner.store.lock():
        recovered = runner._recover()
        report['recovered'] = [(a['call_id'],a['attempt_id'],a['state']) for a in recovered]
        for attempt in recovered:
            if on_outcome:
                on_outcome(attempt)
        if any(a.get('possibly_billed') or a.get('fatal') for a in recovered):
            raise RunError("Recovery found an ambiguous or fatal attempt; no new verification sent")
        runner._ensure_fresh_inputs([c for c in call_ids if runner._runnable(c)])
        chosen, skipped = runner._select(call_ids, retry_rejected=False, reopen_call_ids=set(call_ids) if reopen_reason else set())
        report['skipped'] = skipped
        if not chosen:
            return report
        transport = runner.transport()
        runner._paid_guard(chosen, transport)
        # Build contexts on the parent thread before workers share the immutable requests.
        for call in chosen:
            runner.builder.context(call)
        waiting = iter(chosen)
        active = {}

        def send(attempt, body):
            response = runner._send(attempt, transport, body)  # existing identical-byte capacity policy
            runner._store_response(attempt, response, {'transport':'sync','cohort_workers':max_workers})
            return attempt  # raw response exists before the parent can adjudicate

        def start_next(pool):
            call = next(waiting, None)
            if call is None:
                return False
            preview = runner.builder.build(call,'A000')  # verifier body contains no attempt ID
            if before_prepare:
                before_prepare(call,preview)
            previous = runner.call_state(call)
            reopen_of = next((a['attempt_id'] for a in reversed(previous['attempts']) if a['state']=='ACCEPTED'),None) if reopen_reason else None
            attempt,body = runner._new_attempt(call,'sync',reopen_reason=reopen_reason,reopen_of=reopen_of)
            if on_prepared:
                on_prepared(attempt,body)
            active[pool.submit(send,attempt,body)] = attempt
            return True

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            try:
                for _ in range(max_workers):
                    if not start_next(pool):
                        break
            except Exception as error:
                report['stopped'] = str(error)  # drain anything already sent before returning
            while active:
                done,_ = wait(active,return_when=FIRST_COMPLETED)
                for future in done:
                    attempt = active.pop(future)
                    try:
                        future.result()
                        attempt = runner._adjudicate(attempt)
                    except CapacityError as error:
                        attempt['estimated_cost_usd'] = 0.0  # definite pre-processing capacity rejections
                        runner._close(attempt,'TRANSPORT_FAILED',reason=f'capacity unavailable: {error}',possibly_billed=False)
                        report['stopped'] = str(error)
                    except AmbiguousError as error:
                        attempt['ambiguous'] = True
                        runner._close(attempt,'TRANSPORT_FAILED',reason=f'outcome unknown: {error}',possibly_billed=True)
                        report['stopped'] = str(error)
                    except TransportError as error:
                        if attempt['state'] not in {'TRANSPORT_FAILED','REJECTED','STALE_UPSTREAM'}:
                            if error.fatal:
                                attempt['estimated_cost_usd'] = 0.0  # definitive pre-processing 4xx
                            runner._fail_transport(attempt,error,possibly_billed=not error.fatal)
                        report['stopped'] = str(error)
                    except Exception as error:
                        # A durable RESPONDED attempt is recovered later without another send;
                        # an interrupted PREPARED attempt stays ambiguous under existing recovery.
                        report['stopped'] = f'{type(error).__name__}: {error}'
                    report['attempts'].append(attempt)
                    verdict = None
                    if attempt['state']=='ACCEPTED':
                        from fyp_evaluation.call_specs.compile import parse_json_text
                        artifact = parse_json_text((runner.root/runner.spec.artifact_path(attempt['call_id'])).read_text(encoding='utf-8'))
                        verdict = artifact['verdict']
                    if attempt['state']!='ACCEPTED' or verdict!='VALID':
                        report['stopped'] = report['stopped'] or f"{attempt['call_id']}: {attempt['state']} {verdict}"
                    if on_outcome:
                        on_outcome(attempt)
                if not report['stopped']:
                    try:
                        while len(active)<max_workers and start_next(pool):
                            pass
                    except Exception as error:
                        report['stopped'] = str(error)
    return report
