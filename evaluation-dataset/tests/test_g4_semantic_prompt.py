"""Only new G4 system guidance changes; existing contracts, briefs and old requests stay intact."""
import pytest
from fyp_evaluation.check_plans.config import PROMPT_DIR,prompt_template
from fyp_evaluation.check_plans.request import RequestBuilder
from g4_multilingual_support import plan_inputs,SCHEMAS


@pytest.mark.parametrize('call',['CALL_122','CALL_156'])
def test_v4_adds_guidance_without_changing_brief_or_model_contract(call):
    inputs=plan_inputs(call_ids=[call])
    current=RequestBuilder(inputs,SCHEMAS)
    old=RequestBuilder(inputs,SCHEMAS,prompt_template_name='check_plan.v3.md')
    assert prompt_template(inputs.documents)=='check_plan.v4.md'
    body=current.build(call,'A002'); previous=old.build(call,'A002')
    assert 'template v4' in body['input'][0]['content']
    assert 'K01 and K02 do not count' in body['input'][0]['content']
    assert 'not merely why an execution method' in body['input'][0]['content']
    body['input'][0]=previous['input'][0]
    assert body==previous
    assert current.response_schema==old.response_schema


def test_v3_instructions_are_retained_except_for_header_and_two_additions():
    old=(PROMPT_DIR/'check_plan.v3.md').read_text(encoding='utf-8')
    new=(PROMPT_DIR/'check_plan.v4.md').read_text(encoding='utf-8')
    additions=[line for line in new.splitlines() if line.startswith('For EC_ONE_FIELD_READBACK_ERROR:') or line.startswith('For C07 PASS:')]
    assert len(additions)==2
    for line in additions:new=new.replace('\n'+line+'\n','\n',1)
    new=new.replace('template v4, multilingual releases','template v3, multilingual releases',1)
    assert [line for line in new.splitlines() if line] == [line for line in old.splitlines() if line]
