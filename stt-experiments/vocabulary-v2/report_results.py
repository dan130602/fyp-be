import json,sys
from pathlib import Path
from collections import defaultdict
ROOT=Path('C:/Users/user/Documents/fyp-working/fyp-be-latest')
sys.path.insert(0,str(ROOT/'evaluation-engine'))
from transcribe_eval.export import _language
DATA=ROOT/'evaluation-dataset/datasets/advisory_execution_multilingual_v1_development'
TAG='20261107t175401z-bfb58069'
runids=[f'stt-vocab-broader-{label}-{TAG}' for label in ['baseline','candidate']]
runs=[ROOT/'evaluation-engine/runs'/rid for rid in runids]
meta=[json.loads((r/'metadata.json').read_text()) for r in runs]
assert all(m['status']=='completed' and len(m['calls'])==16 for m in meta)
assert {c['call_id']:c['audio_sha256'] for c in meta[0]['calls']}=={c['call_id']:c['audio_sha256'] for c in meta[1]['calls']}
scores=[json.loads((r/'scores/summary.json').read_text()) for r in runs]
assert scores[0]['normalizer_version']==scores[1]['normalizer_version']=='1.4.0'
rows=[{c['call_id']:c for c in map(json.loads,(r/'scores/calls.jsonl').read_text(encoding='utf-8').splitlines())} for r in runs]
def group(rs):
    result={}
    for language in ['ENGLISH','MANDARIN','EN_ZH_MIXED','SINGLISH','ALL']:
        selected=[r for cid,r in rs.items() if language=='ALL' or _language(DATA,cid)==language]
        result[language]={'calls':len(selected)}
        for field in ['wer','entity_wer','speaker']:
            keys=['eligible_items','correct_items'] if field=='speaker' else ['reference_words','errors','substitutions','deletions','insertions']
            result[language][field]={k:sum(c[field][k] for c in selected) for k in keys}
    return result
groups=[group(r) for r in rows]
def rate(m,field):
    x=m[field]
    return x['correct_items']/x['eligible_items'] if field=='speaker' else x['errors']/x['reference_words']
gates={
    'Overall WER improves':groups[1]['ALL']['wer']['errors']<groups[0]['ALL']['wer']['errors'],
    'Overall entity errors do not increase':groups[1]['ALL']['entity_wer']['errors']<=groups[0]['ALL']['entity_wer']['errors'],
    'No language WER or entity regression':all(groups[1][l][f]['errors']<=groups[0][l][f]['errors'] for l in groups[0] if l!='ALL' for f in ['wer','entity_wer']),
    'Speaker attribution drops no more than 0.2 percentage points':all(rate(groups[1][l],'speaker')>=rate(groups[0][l],'speaker')-0.002 for l in groups[0])}
lines=['# Broader development vocabulary trial','',
'Expanded each language vocabulary from 11 to 33 finance terms and tested a frozen, stratified random sample of 16 additional development calls (four per language). Same audio hashes, handler and scoring rules across both runs. No benchmark text used to choose vocabulary terms.','',
'## Decision','',('PASS: eligible for a full development comparison.' if all(gates.values()) else 'DO NOT PROMOTE: the candidate failed the predeclared gate. Do not spend on a 175-call candidate run or deploy this vocabulary.'),'',
'## Results','','| Language | Calls | Baseline WER | Candidate WER | Baseline entity WER | Candidate entity WER | Baseline speaker | Candidate speaker |','|---|---:|---:|---:|---:|---:|---:|---:|']
for l in groups[0]:
    cells=[f'{rate(groups[i][l],field)*100:.2f}%' for field in ['wer','entity_wer','speaker'] for i in [0,1]]
    lines.append('| '+l+' | '+str(groups[0][l]['calls'])+' | '+' | '.join(cells)+' |')
lines+=['','## Error counts','','| Language | Baseline S / D / I | Candidate S / D / I | WER errors change | Entity errors change |','|---|---:|---:|---:|---:|']
for l in groups[0]:
    counts=[' / '.join(str(g[l]['wer'][k]) for k in ['substitutions','deletions','insertions']) for g in groups]
    delta=lambda f: groups[1][l][f]['errors']-groups[0][l][f]['errors']
    lines.append(f'| {l} | {counts[0]} | {counts[1]} | {delta("wer"):+d} | {delta("entity_wer"):+d} |')
lines+=['','Positive change means more errors. Rates use pooled error counts, not averages of call percentages.','','## Paired call changes','','| Call | Language | Baseline errors | Candidate errors | Entity errors baseline / candidate |','|---|---|---:|---:|---:|']
for cid in rows[0]:
    a,b=rows[0][cid],rows[1][cid]
    lines.append(f'| {cid} | {_language(DATA,cid)} | {a["wer"]["errors"]} | {b["wer"]["errors"]} | {a["entity_wer"]["errors"]} / {b["entity_wer"]["errors"]} |')
lines+=['','## Promotion gates','']+[f'- {k}: {"PASS" if v else "FAIL"}' for k,v in gates.items()]
lines+=['','## Setup and evidence','',
'- 98.9 audio minutes per configuration; 32 jobs total, approximately 197.8 billable audio minutes. Billing amount is not verified.',
'- Seed 20261108. Excluded the earlier four-call canary. Selection used language metadata only for stratification, never for inference routing. The remaining development calls and held-out data were not used for this experiment.',
'- Effective settings: TRANSCRIBE_LANGUAGE_OPTIONS=en-US,zh-CN; TRANSCRIBE_LANGUAGE_ID_MODE=multiple; TRANSCRIBE_MAX_SPEAKER_LABELS=2. Baseline unsets TRANSCRIBE_VOCABULARIES. Candidate uses en-US=clearcall-v2-en-us-'+TAG+',zh-CN=clearcall-v2-zh-cn-'+TAG+'. No custom language model.',
'- The runner verifies actual AWS job parameters and paired audio hashes. Normalizer 1.4.0, unchanged entity definitions, no reviewed domain-term list. Speaker rates use eligible pronunciation items; denominators can differ between model outputs.',
'- 46 production handler tests passed. Vocabulary tables, term provenance and frozen protocol are under stt-experiments/vocabulary-v2. Raw outputs remain under git-ignored evaluation-engine/runs.',
'- No production deployment, dataset/scorer edits, held-out evaluation, publishing, commit or push. New AWS vocabulary resources and normal evaluation outputs remain available for audit.',
'','## Error-analysis context','',
'Only the earlier four-call detailed runs were available locally; the 175-call screenshot is aggregate evidence, not a downloadable confusion inventory. Earlier errors include language-switch omissions, Chinese homophones and names. Those cannot be reliably solved by adding a list of benchmark answers. The wider sample above provides fresh evidence about how far generic domain vocabulary helps.',
'','## Source provenance','',
'Added 22 concepts per language from [MoneySense bond guidance](https://www.moneysense.gov.sg/understanding-bonds/), [MoneySense ETF comparison guidance](https://www.moneysense.gov.sg/investments/choose-etfs-same-index/), and [SGX trading guidance](https://www.sgx.com/securities/trading). Chinese renderings were drafted for this experiment; they are not claimed to be official translations. The original 11 concepts retain their existing provenance in stt-experiments/README.md.',
'','Run IDs:','']+['- `'+rid+'`' for rid in runids]
lines += ['', '## Diagnosis and next decision', '',
    'The broader baseline contains 240 substitutions, 131 deletions and 34 insertions. Mixed-language calls account for 95 of the 131 deletions (72.5%); the candidate increases mixed-language deletions to 96. This is the clearest priority for the next architectural experiment.', '',
    'Paired model-output inspection found English word-boundary regressions, a two-character Mandarin omission, and an English word changing grammatical form. Capitalization and multiword item packaging also changed, but those formatting changes alone do not explain the scored regression. Two calls improved by one error each, four worsened, and ten were unchanged. Entity-error counts were unchanged on every call.', '',
    'Reject the 33-term candidate. The earlier 11-term candidate has only a one-error gain on four calls, so it is also not validated for production. Retain the current deployed configuration; the opt-in vocabulary code remains available for future controlled experiments.', '',
    'The next research direction is audio-only language-switch handling: use timing and language evidence from audio/model output to identify omitted or uncertain spans, then evaluate a segment-level transcription candidate against this frozen baseline. Never use expected text or benchmark labels to route inference. This needs a separately implemented and tested segmentation/reassembly pipeline; it is not implemented or claimed as an improvement here. A full 175-call run and owner deployment remain conditional on a candidate passing the development gate.', '',
    'The runner supplied settings only to its own process. No evaluation-engine/.env exists in this checkout, so the rejected vocabulary was not saved as a persistent default. Final git diff checks passed, with no tracked scorer or dataset modifications.']
report='\n'.join(lines)+'\n'
(ROOT/'stt-experiments/vocabulary-v2/RESULTS.md').write_text(report,encoding='utf-8')
out=Path('C:/Users/user/Documents/Codex/2026-11-06/i-ha/outputs')
(out/'STT-Broader-Vocabulary-Trial.md').write_text(report,encoding='utf-8')
(ROOT/'stt-experiments/vocabulary-v2/aggregate-results.json').write_text(json.dumps(dict(run_ids=runids,groups=groups,gates=gates),indent=2),encoding='utf-8')
print(report)
