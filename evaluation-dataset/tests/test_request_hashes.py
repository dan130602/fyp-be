"""Request-hash regression over the committed G3/G4 attempts (read-only; no network).

Every committed attempt is rebuilt with ITS OWN recorded attempt ID (the attempt ID is part of the OpenAI request metadata,
so a dry run for the next attempt can never reproduce a committed hash) and must hash to the committed `request_sha256`.

Exactly seven G4 attempts are pinned exceptions: the six calls with a READBACK event were requested before the approved
CALL_037 brief fix that added `form_choices` and a longer `readback.note` to the READBACK event brief (CALL_037's two
rejected attempts included). Their audit `request.json` files still hash to the committed values; only today's rebuild
differs. The test pins today's rebuilt hash for each, so any further unintended change to those requests is caught too.
Historical committed hashes are never altered.
"""

import pytest

from fyp_evaluation.call_configs.run import ConfigRunner
from fyp_evaluation.call_specs.compile import text_hash
from fyp_evaluation.check_plans.run import CheckPlanRunner
from fyp_evaluation.llm.store import canonical_request_text
from g5_support import RELEASE, SCHEMAS

READBACK_BRIEF_FIX_EXCEPTIONS = {
    ("CALL_003", "A001"): "5c6dc1f837c669609899bc3860cffea28229b368fe7482db7a22783be316cb58",
    ("CALL_017", "A001"): "3e41723c28d016c6add18e0712ba9fefcb0446df6894933d6148e56e3ec3006e",
    ("CALL_034", "A001"): "ef849e07249f0c4e2e7c569b34f89c28910bc6d4001d8b96fd39e6e8b10dc4f5",
    ("CALL_037", "A001"): "23e8cfdebe93284cd91deb061be8269e59e5944c8b1b451973c80ae2a5718ab1",
    ("CALL_037", "A002"): "a57569d71d5c6f50d25fc8d2009a82dff946decc65d91311ccd47f5258372247",
    ("CALL_040", "A001"): "88eb17f815b0bcbd25e8c3da83939b173963158e23c2eaccb241e47d1f422b0b",
    ("CALL_043", "A001"): "59ef6054bc693558fb22adb2c8c4be810feabbad3e07e29648d2cc1e481c78bf",
}


def _rebuilt(runner_class):
    runner = runner_class(RELEASE, schema_dir=SCHEMAS)
    result = {}
    for call_id in runner.inputs.call_ids:
        prov, problem = runner._prov_state(call_id)
        assert problem is None, (call_id, problem)
        for attempt in prov["attempts"]:
            rebuilt = text_hash(canonical_request_text(runner.builder.build(call_id, attempt["attempt_id"])))
            result[(call_id, attempt["attempt_id"])] = (attempt["request_sha256"], rebuilt)
    return result


@pytest.mark.parametrize("runner_class, exceptions", [(ConfigRunner, {}), (CheckPlanRunner, READBACK_BRIEF_FIX_EXCEPTIONS)],
                         ids=["G3", "G4"])
def test_committed_requests_rebuild_to_their_recorded_hashes(runner_class, exceptions):
    rebuilt = _rebuilt(runner_class)
    assert rebuilt, "no committed attempts found"
    mismatched = {key for key, (recorded, now) in rebuilt.items() if recorded != now}
    assert mismatched == set(exceptions), sorted(mismatched ^ set(exceptions))
    for key, pinned in exceptions.items():
        assert rebuilt[key][1] == pinned, f"{key}: today's rebuilt request changed again"
