"""Hypothesis-only label recovery for collapsed Qwen timestamp points.

No reference text/times, fabricated durations, interpolation, ASR or training.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'work/meralion-alignment-pilot'
sys.path.insert(0,str(BUNDLE))
from alignment_core import canonical,chunk_items,document

CONFIG={'method':'guarded-collapsed-point-v1','window_each_side_seconds':.2,
        'maximum_anchor_distance_seconds':2.,'fabricate_word_durations':False,
        'require_both_anchors':True,'require_continuous_same_speaker_window':True}


def stable_window(segments,label,start,end):
    cursor=start
    for row in segments:
        left,right=max(start,row['start']),min(end,row['end'])
        if right<=left: continue
        if row['speaker']!=label or left>cursor+1e-7: return False
        cursor=max(cursor,right)
    return cursor>=end-1e-7


def recover_chunk(text,units,offset,duration,segments):
    items,details=chunk_items(text,units,offset,duration,segments)
    original=deepcopy(items)
    source=[(char,index) for index,original_char in enumerate(text) for char in canonical(original_char)]
    lexical=[(unit,canonical(unit['text'])) for unit in units if canonical(unit['text'])]
    if ''.join(char for char,_ in source)!=''.join(word for _,word in lexical):
        return items,details,[]
    owners=[set() for _ in text]
    cursor=0
    for index,(_,word) in enumerate(lexical):
        for _,position in source[cursor:cursor+len(word)]: owners[position].add(index)
        cursor+=len(word)
    candidates={}
    for index,(item,detail) in enumerate(zip(items,details)):
        if item['type']!='pronunciation' or detail.get('reason')!='invalid_alignment_time': continue
        used={owner for position in range(detail['start_char'],detail['end_char']) for owner in owners[position]}
        if len(used)!=1: continue
        unit=lexical[next(iter(used))][0]
        begin,end=float(unit['start_time']),float(unit['end_time'])
        if math.isfinite(begin) and begin==end and 0<=begin<duration:
            candidates[index]=offset+begin

    def anchor(index,step):
        for position in range(index+step,len(items) if step>0 else -1,step):
            item=original[position]
            if item['type']=='punctuation': continue
            if item.get('speaker_label') and 'start_time' in item and 'end_time' in item:
                return position,item
            # Cross only other zero-point candidates, never an unrelated failure.
            if position not in candidates: return None
        return None

    recovered=[]
    margin=CONFIG['window_each_side_seconds']
    maximum=CONFIG['maximum_anchor_distance_seconds']
    for index,point in candidates.items():
        left,right=anchor(index,-1),anchor(index,1)
        if left is None or right is None: continue
        li,previous=left
        ri,following=right
        label=previous['speaker_label']
        if label!=following['speaker_label']: continue
        if not float(previous['start_time'])<=point<=float(following['end_time']): continue
        if max(abs(point-float(previous['end_time'])),abs(float(following['start_time'])-point))>maximum: continue
        if point-margin<offset or point+margin>offset+duration: continue
        if not stable_window(segments,label,point-margin,point+margin): continue
        items[index]['speaker_label']=label
        recovered.append(dict(item=index,point_seconds=point,speaker_label=label,
                              left_anchor_item=li,right_anchor_item=ri))
        details[index]['original_reason']=details[index].get('reason')
        details[index]['reason']=None
        details[index]['speaker_source']=CONFIG['method']
    # The only allowed item mutation is addition of a predicted speaker label.
    for before,after in zip(original,items):
        left,right=deepcopy(before),deepcopy(after)
        left.pop('speaker_label',None); right.pop('speaker_label',None)
        assert left==right
        if before.get('speaker_label'): assert before==after
    return items,details,recovered


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    entries=read(BUNDLE/'pilot_manifest.json')['calls']
    assert {p.stem for p in args.source.glob('CALL_*.json')}=={e['call_id'] for e in entries}
    prepared=[]
    for entry in entries:
        cid=entry['call_id']
        source_path=BUNDLE/'transcripts'/f'{cid}.json'
        speaker_path=BUNDLE/'speakers'/f'{cid}.json'
        assert hashlib.sha256(source_path.read_bytes()).hexdigest()==entry['transcript_sha256']
        assert hashlib.sha256(speaker_path.read_bytes()).hexdigest()==entry['speaker_sha256']
        source,speaker,row=read(source_path),read(speaker_path),read(args.source/f'{cid}.json')
        assert row['call_id']==cid and source['settings']==row['settings']
        assert row['source_transcript_sha256']==hashlib.sha256(source['transcript'].encode()).hexdigest()
        predictions={c['chunk']:c for c in row['alignment']}
        items,audit,recovered=[],[],[]
        base_items=[]
        for index,chunk in enumerate(source['chunks'],1):
            if not chunk['text']: continue
            aligned=predictions.pop(index)
            left,right=chunk['start_sample'],chunk['end_sample']
            assert (left,right)==(aligned['start_sample'],aligned['end_sample'])
            arguments=(chunk['text'],aligned['units'],left/16000,(right-left)/16000,speaker['exclusive_speaker_diarization'])
            baseline,_=chunk_items(*arguments)
            base_items.extend(baseline)
            added,details,changed=recover_chunk(*arguments)
            for detail in details: detail['chunk']=index
            for record in changed:
                record['chunk']=index
                record['global_item']=len(items)+record['item']
            items.extend(added); audit.extend(details); recovered.extend(changed)
        assert not predictions
        assert document(source['transcript'],base_items)==row['with_speakers'],cid
        before=deepcopy(row['with_speakers'])
        after=deepcopy(document(source['transcript'],items))
        for obj in (before,after):
            for item in obj['results']['items']: item.pop('speaker_label',None)
        assert before==after==row['without_speakers'],cid
        result=deepcopy(row)
        result['with_speakers']=document(source['transcript'],items)
        result['audit']=audit
        result['speaker_recovery']=dict(config=CONFIG,recovered_items=recovered,
            original_coverage=row['coverage'],source_alignment_sha256=hashlib.sha256((args.source/f'{cid}.json').read_bytes()).hexdigest())
        pronunciation=[i for i in items if i['type']=='pronunciation']
        result['coverage']=dict(pronunciation_items=len(pronunciation),
            timed_items=sum('start_time' in i for i in pronunciation),
            speaker_labeled_items=sum('speaker_label' in i for i in pronunciation),
            recovered_speaker_items=len(recovered),
            unassigned_reasons=dict(Counter(a.get('reason') for i,a in zip(items,audit) if i['type']=='pronunciation' and not i.get('speaker_label'))))
        assert result['coverage']['speaker_labeled_items']==row['coverage']['speaker_labeled_items']+len(recovered)
        assert result['coverage']['timed_items']==row['coverage']['timed_items']
        assert not (args.output/f'{cid}.json').exists(),'Existing output; choose a new directory'
        prepared.append(result)
    args.output.mkdir(parents=True,exist_ok=True)
    for row in prepared:
        with (args.output/f'{row["call_id"]}.json').open('x',encoding='utf-8') as handle:
            json.dump(row,handle,ensure_ascii=False,indent=2,allow_nan=False)
        print(row['call_id'],row['coverage'])


if __name__=='__main__': main()
