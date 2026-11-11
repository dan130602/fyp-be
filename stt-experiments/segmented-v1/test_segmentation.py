from array import array
from copy import deepcopy
import hashlib
import wave
import pytest
from segmentation import split_pcm, assemble, speaker_for


def word(text, start, end, speaker='local-0'):
    return dict(type='pronunciation',start_time=str(start),end_time=str(end),speaker_label=speaker,alternatives=[dict(content=text,confidence='0.9')])


def test_pcm_bytes_original_and_complete_cores(tmp_path):
    source=tmp_path/'source.wav'
    rate=100
    pcm=array('h', [0,100,-100,200]*6000).tobytes()
    with wave.open(str(source),'wb') as w:
        w.setparams((1,2,rate,0,'NONE','not compressed')); w.writeframes(pcm)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    chunks=split_pcm(source,tmp_path/'chunks')
    assert chunks[0]['core_start']==0 and chunks[-1]['core_end']==240
    assert all(a['core_end']==b['core_start'] for a,b in zip(chunks,chunks[1:]))
    assert all(c['core_end']-c['core_start']>=30 for c in chunks)
    for c in chunks:
        with wave.open(c['path'],'rb') as w:
            assert w.getframerate()==rate
            assert w.readframes(w.getnframes())==pcm[c['start_frame']*2:c['end_frame']*2]
    assert before==hashlib.sha256(source.read_bytes()).hexdigest()


def test_overlap_ownership_global_timing_and_speaker_abstention():
    chunks=[dict(core_start=0,core_end=45,offset=0,final=False),dict(core_start=45,core_end=90,offset=44,final=True)]
    docs=[{'results':{'items':[word('first',43,44),word('second',45.1,45.5)]}},
          {'results':{'items':[word('first',-1,0),word('second',1.1,1.5),word('uncovered',10,11)]}}]
    full={'results':{'items':[word('x',43,44,'global-1'),word('y',45,46,'global-0')]}}
    old=deepcopy(docs)
    out=assemble(chunks,docs,full)['results']['items']
    assert [x['alternatives'][0]['content'] for x in out]==['first','second','uncovered']
    assert out[1]['start_time']=='45.1' and out[1]['speaker_label']=='global-0'
    assert 'speaker_label' not in out[2]
    assert docs==old


def test_missing_segment_fails_and_weak_speaker_abstains():
    with pytest.raises(ValueError): assemble([{}],[],{})
    assert speaker_for(0,1,[word('x',0,0.1,'a')]) is None


def test_punctuation_and_real_repetition_retained():
    p={'type':'punctuation','alternatives':[{'content':'.'}]}
    doc={'results':{'items':[word('no',1,2),word('no',3,4),p]}}
    result=assemble([dict(core_start=0,core_end=10,offset=0,final=True)],[doc],doc)
    assert result['results']['transcripts'][0]['transcript']=='no no.'
