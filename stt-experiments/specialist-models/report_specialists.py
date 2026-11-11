import json
from pathlib import Path
ROOT=Path('C:/Users/user/Documents/fyp-working/fyp-be-latest')
RUNS=ROOT/'evaluation-engine/runs'
BASE='stt-vocab-broader-baseline-20261107t175401z-bfb58069'
VOCAB='stt-vocab-original16-candidate-20261108t100155z-266e558b'
WHISPER='whisper-small-en-singlish-20261108T100344Z'
def load(rid): return {c['call_id']:c for c in map(json.loads,(RUNS/rid/'scores/calls.jsonl').read_text(encoding='utf-8').splitlines())}
a,b,w=load(BASE),load(VOCAB),load(WHISPER)
assert len(a)==len(b)==16 and len(w)==4
def group(rows,ids):
    return {f:{k:sum(rows[i][f][k] for i in ids) for k in ks} for f,ks in [('wer',['errors','reference_words','substitutions','deletions','insertions']),('entity_wer',['errors','reference_words']),('speaker',['correct_items','eligible_items'])]}
def pct(r,f):
    n=r[f];return f"{100*(n['correct_items']/n['eligible_items'] if f=='speaker' else n['errors']/n['reference_words']):.2f}%"
groups={'English':['CALL_071','CALL_090','CALL_091','CALL_139'],'Mandarin':['CALL_003','CALL_123','CALL_177','CALL_244'],'Mixed':['CALL_001','CALL_035','CALL_041','CALL_209'],'Singlish':['CALL_146','CALL_184','CALL_198','CALL_235'],'Overall':list(a)}
lines=['# Original vocabulary validation and specialist-model research','','## Original 11-term vocabulary: broader validation','',
'This tests the original vocabulary that reduced errors from 47 to 46 on the initial four calls. It is distinct from the rejected 33-term expansion. Sixteen additional development calls were evaluated against the existing matched baseline; only the candidate required fresh AWS jobs (98.9 audio minutes).','',
'| Language | Baseline WER | Original vocabulary WER | Baseline entity WER | Vocabulary entity WER | Baseline speaker | Vocabulary speaker |','|---|---:|---:|---:|---:|---:|---:|']
for language,ids in groups.items():
    x,y=group(a,ids),group(b,ids)
    lines.append('| '+language+' | '+' | '.join(pct(r,f) for f in ['wer','entity_wer','speaker'] for r in [x,y])+' |')
x,y=group(a,a),group(b,b)
lines+=['',f"Total errors: {x['wer']['errors']} to {y['wer']['errors']} out of {x['wer']['reference_words']}. Entity errors: {x['entity_wer']['errors']} to {y['entity_wer']['errors']} out of {x['entity_wer']['reference_words']}.",
'The initial one-error gain did not generalize to this broader sample. Do not promote the vocabulary or spend on a full 175-call candidate run based on this evidence. This small difference does not establish statistical superiority or inferiority; it fails the practical improvement gate.',
'','## Whisper small as an English-configured Singlish specialist','',
'Four predeclared Singlish calls, compared with the same Transcribe calls. This is a specialist feasibility experiment, not an automatic routing evaluation. Whisper receives the same fixed English setting for every file; benchmark labels/answers are not read by its inference path. The cohort is known Singlish for evaluation, so results cannot establish how to identify Singlish in production.','',
'| Model | WER | Entity WER | Speaker attribution |','|---|---:|---:|---:|']
for name,rs in [('Transcribe baseline',a),('Whisper small, English setting',w)]:
    g=group(rs,w)
    lines.append(f'| {name} | {pct(g,"wer")} | {pct(g,"entity_wer")} | '+('Unavailable' if rs is w else pct(g,'speaker'))+' |')
lines+=['','| Call | Transcribe WER | Whisper WER | Transcribe entity WER | Whisper entity WER |','|---|---:|---:|---:|---:|']
for cid in w:lines.append('| '+cid+' | '+' | '.join(pct(rs[cid],f) for f in ['wer','entity_wer'] for rs in [a,w])+' |')
wm=json.loads((RUNS/WHISPER/'metadata.json').read_text())
seconds=sum(c['inference_seconds'] for c in wm['calls']);audio=sum(c['audio_seconds'] for c in wm['calls'])
g0,g1=group(a,w),group(w,w)
lines+=['',f"Whisper errors: {g1['wer']['errors']}/{g1['wer']['reference_words']} versus Transcribe {g0['wer']['errors']}/{g0['wer']['reference_words']}; entity errors {g1['entity_wer']['errors']}/{g1['entity_wer']['reference_words']} versus {g0['entity_wer']['errors']}/{g0['entity_wer']['reference_words']}.",
('Whisper passes the lexical/entity feasibility check; broader testing and diarization/router validation are still required.' if g1['wer']['errors']<g0['wer']['errors'] and g1['entity_wer']['errors']<=g0['entity_wer']['errors'] else 'Whisper does not establish a better Singlish specialist on this sample. Retain Transcribe.'),
f'CPU inference took {seconds/60:.2f} minutes for {audio/60:.2f} audio minutes, excluding download/loading/scoring. No paid API or GPU was used for Whisper.',
'- Systran/faster-whisper-small revision 536b0662742c02347bc0e980a01041f333bce120; CPU INT8, four threads, beam size 5, language en, task transcribe, word timestamps enabled, VAD disabled; no prompt or vocabulary.',
'- Both model size and language setting differ from the earlier base-model experiment, which also used different calls. Do not attribute changes to model size alone or compare those cohort percentages directly.',
'- No speaker labels or calibrated confidence were invented. The unchanged scorer reports zero for absent speaker labels, but speaker attribution is unavailable for this standalone Whisper pipeline.',
'- Adapter checks covered decimals, punctuation, repeated words and absent speaker/confidence. Source audio hashes match the baseline and remain unchanged.',
'','## Research findings','',
'| Candidate | Why consider it | Status here |','|---|---|---|',
'| Transcribe | Existing multilingual and diarization integration; strongest measured default so far | Baseline and vocabulary experiments measured |',
'| Whisper small | 244M parameters versus base 74M; local CPU inference is feasible | Measured above; no automatic Singlish routing |',
'| Whisper large-v3 / turbo | Larger multilingual candidates with different speed/accuracy tradeoffs | Not measured; larger runtime budget and diarization integration needed |',
'| MERaLiON-3-3B-ASR | Authors explicitly target Singapore/regional speech and code-switching | Researched, not run; no verified runtime or API access available |',
'| Qwen3-ASR 0.6B / 1.7B | English/Chinese support and language identification; separate forced aligner provides timestamps | Researched, not run; needs its own runtime/adapter validation |',
'',
'MERaLiON has distinct hosted API and open-weight offerings. The card describes self-hosting the open model on a single 80 GB GPU; that is not established here as a minimum memory requirement or proof CPU inference is impossible. Its published benchmark results are not comparable to these synthetic financial calls. The hosted API advertises timestamps/diarization; those capabilities should not be assumed for the open-weight text-only wrapper.',
'',
'Qwen documents 30 languages and 22 Chinese dialects, with a separate forced-alignment model. This is a research candidate, not evidence of superior Singlish performance. The documented CUDA examples and FlashAttention build guidance are not a verified CPU benchmark for this laptop.',
'','## What a valid multi-model design requires','',
'1. Establish a specialist advantage on a development cohort, including entity/number errors, before building routing around it.',
'2. Evaluate proper English and Mandarin alongside Singlish and mixed calls. English language identification alone cannot distinguish proper English from Singlish; mixed calls may switch within an utterance.',
'3. Route using audio-derived evidence or independently supplied operational metadata, never benchmark labels, IDs or expected answers. Low-confidence or unsupported routes should use the validated default.',
'4. Measure the complete router-plus-recognizer pipeline on a separate development validation slice: route confusion matrix, WER, entity WER, speaker accuracy, latency and cost. A table that selects the best model using known dataset labels is an oracle comparison, not deployable accuracy.',
'5. Preserve speaker identity and timing across any fallback pass. Do not select the winning transcript using reference answers or blindly compare confidence numbers from different models.',
'6. Only after a candidate passes these checks, run all 175 development calls; the owner runs the held-out set and deploys. No full run or deployment is justified by the unsuccessful trials alone.',
'','## Sources','',
'- [OpenAI Whisper models and language behavior](https://github.com/openai/whisper)',
'- [Faster-whisper CPU inference](https://github.com/SYSTRAN/faster-whisper)',
'- [MERaLiON official model card](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR)',
'- [Qwen official model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)',
'','## Audit','',
'Vocabulary settings: TRANSCRIBE_LANGUAGE_OPTIONS=en-US,zh-CN; TRANSCRIBE_LANGUAGE_ID_MODE=multiple; TRANSCRIBE_MAX_SPEAKER_LABELS=2; TRANSCRIBE_VOCABULARIES=en-US=clearcall-canary-en-us-20261107t171418z-d03fd238,zh-CN=clearcall-canary-zh-cn-20261107t171418z-d03fd238. Actual AWS settings and audio hashes were verified. Baseline omits vocabulary. Normalizer/scorer and entity definitions unchanged. No held-out data, production deployment, publishing, commits or pushes. No persistent environment setting changed.',
'','Run IDs:',f'- `{BASE}`',f'- `{VOCAB}`',f'- `{WHISPER}`','']
report='\n'.join(lines)
out=ROOT/'stt-experiments/specialist-models';out.mkdir(exist_ok=True)
(out/'RESULTS_AND_RESEARCH.md').write_text(report,encoding='utf-8')
Path('outputs/STT-Specialist-Models-and-Vocabulary.md').write_text(report,encoding='utf-8')
print(report)
