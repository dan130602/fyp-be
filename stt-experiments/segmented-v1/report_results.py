import json
from pathlib import Path
ROOT=Path('C:/Users/user/Documents/fyp-working/fyp-be-latest')
RUN='stt-segmented-20261108t053907z-b3f0bb6f'
BASE='stt-vocab-broader-baseline-20261107t175401z-bfb58069'
run=ROOT/'evaluation-engine/runs'/RUN
plan=json.loads((run/'segment_plan.json').read_text())
ids={c['call_id'] for c in plan['calls']}
def rows(rid):
    return {c['call_id']:c for c in map(json.loads,(ROOT/'evaluation-engine/runs'/rid/'scores/calls.jsonl').read_text(encoding='utf-8').splitlines()) if c['call_id'] in ids}
a,b=rows(BASE),rows(RUN)
assert set(a)==set(b)==ids
def aggregate(rs):
    return {f:{k:sum(r[f][k] for r in rs.values()) for k in keys} for f,keys in [('wer',['errors','reference_words','substitutions','deletions','insertions']),('entity_wer',['errors','reference_words']),('speaker',['correct_items','eligible_items'])]}
x,y=aggregate(a),aggregate(b)
def rate(r,f):
    q=r[f]
    return q['correct_items']/q['eligible_items'] if f=='speaker' else q['errors']/q['reference_words']
passed=y['wer']['errors']<x['wer']['errors'] and y['entity_wer']['errors']<=x['entity_wer']['errors'] and rate(y,'speaker')>=rate(x,'speaker')-0.002
lines=['# Segmented STT trial results','',
('Decision: passed the mixed-language pilot gate; broader multilingual validation is still required.' if passed else 'Decision: failed the pilot gate. Do not deploy or run all 175 calls with this candidate.'),'',
'Four mixed-language development calls; the baseline below is the matching four-call subset, not the entire earlier 16-call run.','','| Metric | Full-call baseline | Segmented candidate |','|---|---:|---:|']
for label,f in [('WER','wer'),('Entity WER','entity_wer'),('Speaker attribution','speaker')]:
    lines.append(f'| {label} | {rate(x,f)*100:.2f}% | {rate(y,f)*100:.2f}% |')
lines+=['',f"Word/character errors: {x['wer']['errors']} → {y['wer']['errors']} / {x['wer']['reference_words']}.",f"Entity errors: {x['entity_wer']['errors']} → {y['entity_wer']['errors']} / {x['entity_wer']['reference_words']}.",
f"Substitutions/deletions/insertions: {x['wer']['substitutions']}/{x['wer']['deletions']}/{x['wer']['insertions']} → {y['wer']['substitutions']}/{y['wer']['deletions']}/{y['wer']['insertions']}.",'',
'| Call | Baseline WER | Candidate WER | Baseline entity WER | Candidate entity WER | Baseline speaker | Candidate speaker |','|---|---:|---:|---:|---:|---:|---:|']
for cid in sorted(ids):
    values=[f'{rate(r[cid],f)*100:.2f}%' for f in ['wer','entity_wer','speaker'] for r in [a,b]]
    lines.append('| '+cid+' | '+' | '.join(values)+' |')
unassigned=total=0
for cid in ids:
    doc=json.loads((run/'raw/transcripts'/f'{cid}.json').read_text(encoding='utf-8'))
    words=[i for i in doc['results']['items'] if i['type']=='pronunciation']
    total+=len(words); unassigned+=sum(not i.get('speaker_label') for i in words)
lines+=['','## Method and limitations','',
f'- 19 segment jobs, {plan["segment_audio_minutes"]:.2f} additional audio minutes. Reused the previous full-call Transcribe output for global speaker evidence; a deployed two-pass system would also pay for that first pass.',
'- Approximately 45-second cores selected at quiet audio boundaries, one second of context each side, timestamp midpoint ownership on reassembly. Exact PCM byte slices; original file hashes verified before and after inference.',
'- Effective settings: TRANSCRIBE_LANGUAGE_OPTIONS=en-US,zh-CN; TRANSCRIBE_LANGUAGE_ID_MODE=multiple; TRANSCRIBE_MAX_SPEAKER_LABELS=2. No vocabulary or custom language model. Actual job settings checked for every segment.',
f'- Full-call speaker evidence is projected by timestamp overlap, with abstention below 50% overlap. {unassigned}/{total} output pronunciation items have no speaker label. Speaker accuracy includes the consequences of abstention; item coverage and eligible denominators can differ between outputs.',
f'- Eligible speaker items: baseline {x["speaker"]["correct_items"]}/{x["speaker"]["eligible_items"]}, candidate {y["speaker"]["correct_items"]}/{y["speaker"]["eligible_items"]}.',
'- Existing scorer and normalizer 1.4.0 unchanged. Four offline tests passed before execution. Reference text and benchmark labels were not used in segmentation, language routing or assembly.',
'- This sample contains only mixed-language calls. English, Mandarin-only and Singlish regression testing has not been performed for this pipeline. No held-out data used.',
'- No production deployment, persistent settings change, publishing, commit or push. Original audio and datasets unchanged. Temporary S3 input clips were removed after successful scoring; normal output transcripts and local audit files remain.',
'','## Run identifiers','',f'- Baseline subset: `{BASE}`',f'- Candidate: `{RUN}`','']
report='\n'.join(lines)
(ROOT/'stt-experiments/segmented-v1/RESULTS.md').write_text(report,encoding='utf-8')
Path('outputs/STT-Segmented-Trial-Results.md').write_text(report,encoding='utf-8')
print(report)
