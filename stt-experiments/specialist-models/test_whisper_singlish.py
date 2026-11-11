from types import SimpleNamespace as N
from whisper_singlish import convert

def test_decimal_punctuation_repetition_and_timing():
    words=[N(word=' 1.25,',start=0,end=1),N(word=' no',start=1,end=2),N(word=' no。',start=2,end=3)]
    result=convert([N(text=' 1.25, no no。',words=words)])['results']
    assert [i['alternatives'][0]['content'] for i in result['items']]==['1.25',',','no','no','。']
    assert result['items'][0]['start_time']=='0' and result['items'][0]['end_time']=='1'
    assert all('speaker_label' not in i and 'confidence' not in i['alternatives'][0] for i in result['items'])

def test_contraction_thousands_and_chinese_characters_preserved():
    result=convert([N(text=" don't 1,000 中文",words=[N(word=w,start=0,end=1) for w in ["don't",'1,000','中文']])])
    assert [i['alternatives'][0]['content'] for i in result['results']['items']]==["don't",'1,000','中文']
