"""Fixed-English Whisper-small diagnostic on a predeclared Singlish cohort, not a router."""
import hashlib,json,re,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/user/Documents/fyp-working/fyp-be-latest')
ENGINE=ROOT/'evaluation-engine'
sys.path.insert(0,str(ENGINE))
DATASET='advisory_execution_multilingual_v1_development'
BASE='stt-vocab-broader-baseline-20261107t175401z-bfb58069'
IDS=['CALL_146','CALL_184','CALL_198','CALL_235']

def convert(segments):
    items=[];texts=[]
    for segment in segments:
        texts.append(segment.text)
        for word in segment.words or []:
            for token in re.findall(r"\w+(?:['’.,]\w+)*|[^\w\s]",word.word,re.UNICODE):
                item={'type':'pronunciation' if any(c.isalnum() for c in token) else 'punctuation','alternatives':[{'content':token}]}
                if item['type']=='pronunciation': item.update(start_time=str(word.start),end_time=str(word.end))
                items.append(item)
    return {'results':{'items':items,'transcripts':[{'transcript':''.join(texts)}]}}

def main():
    from faster_whisper import WhisperModel
    from transcribe_eval.collect import local_audio
    from transcribe_eval.reference import load_terms
    from transcribe_eval.score import score_run
    modelinfo=json.loads(Path('C:/Users/user/Documents/Codex/2026-11-06/i-ha/work/whisper-small-model-info.json').read_text())
    baseline=json.loads((ENGINE/'runs'/BASE/'metadata.json').read_text())
    hashes={c['call_id']:c['audio_sha256'] for c in baseline['calls']}
    run=ENGINE/'runs'/('whisper-small-en-singlish-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    (run/'raw/transcripts').mkdir(parents=True,exist_ok=False)
    metadata=dict(run_id=run.name,dataset_id=DATASET,status='running',calls=[],model=modelinfo,
        settings=dict(device='cpu',compute_type='int8',cpu_threads=4,beam_size=5,language='en',task='transcribe',word_timestamps=True,vad_filter=False),
        limitation='Predeclared Singlish-only English-configured specialist diagnostic. No production router, no diarization, no calibrated AWS-comparable confidence. Never select model by gold at inference.')
    (run/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    model=WhisperModel(modelinfo['path'],device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
    for cid in IDS:
        audio=local_audio(ROOT/'evaluation-dataset/datasets'/DATASET,cid,'auto')
        digest=hashlib.sha256(audio.read_bytes()).hexdigest(); assert digest==hashes[cid]
        print('Starting',cid,flush=True);start=time.monotonic()
        segments,info=model.transcribe(str(audio),language='en',task='transcribe',beam_size=5,word_timestamps=True,vad_filter=False)
        doc=convert(segments)
        relative=f'raw/transcripts/{cid}.json'
        (run/relative).write_text(json.dumps(doc,ensure_ascii=False),encoding='utf-8')
        assert hashlib.sha256(audio.read_bytes()).hexdigest()==digest
        metadata['calls'].append(dict(call_id=cid,status='COMPLETED',transcript_file=relative,audio_sha256=digest,audio_seconds=info.duration,inference_seconds=round(time.monotonic()-start,2)))
        (run/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print('Completed',cid,metadata['calls'][-1]['inference_seconds'],flush=True)
    metadata['status']='completed';(run/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    reviewed,terms=load_terms(ENGINE/'domain_terms.json')
    summary=score_run(run,terms=terms,terms_reviewed=reviewed)
    print(json.dumps({k:summary[k] for k in ['run_id','calls_scored','normalised_wer','entity_wer']}),flush=True)

if __name__=='__main__': main()
