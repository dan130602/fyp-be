from types import SimpleNamespace
from fyp_evaluation.dialogue_verifications.request import RequestBuilder
from g4_multilingual_support import upstream_inputs,SCHEMAS
import g7_support


def test_multilingual_g7_preserves_base_contract_and_extends_authoring_scope():
    inputs=SimpleNamespace(documents=upstream_inputs().documents)
    current=RequestBuilder(inputs,SCHEMAS)
    old=RequestBuilder(inputs,SCHEMAS,prompt_template_name='dialogue_verification.v2.md')
    assert current.response_schema==old.response_schema
    assert 'AUTHORING CONFORMANCE' in current.system_instruction
    assert 'an unchanged gold answer is not itself contradicted' in current.system_instruction
    assert 'unplanned refusal to execute' in current.system_instruction
    assert 'authorized planned instruction' in current.system_instruction
    start=current.system_instruction.index('AUTHORING CONFORMANCE')
    end=current.system_instruction.index('CHECKLIST (frozen)',start)
    assert current.system_instruction[:start]+current.system_instruction[end:]==old.system_instruction
    for builder in [current,old]:builder.brief_text=lambda call:'unchanged fixture brief'
    now=current.build('CALL_001','A001');before=old.build('CALL_001','A001')
    now['system_instruction']=before['system_instruction']
    assert now==before


def test_historical_g7_keeps_its_original_prompt():
    inputs=g7_support.shared_inputs()
    current=RequestBuilder(inputs,SCHEMAS)
    old=RequestBuilder(inputs,SCHEMAS,prompt_template_name='dialogue_verification.v1.md')
    assert current.system_instruction==old.system_instruction
