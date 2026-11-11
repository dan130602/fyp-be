"""Two-pass segmented Transcribe experiment. Plan offline; execute once after login."""
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from segmentation import split_pcm, assemble

ROOT=Path('C:/Users/user/Documents/fyp-working/fyp-be-latest')
ENGINE=ROOT/'evaluation-engine'
sys.path.insert(0,str(ENGINE))
DATASET='advisory_execution_multilingual_v1_development'
BASELINE='stt-vocab-broader-baseline-20261107t175401z-bfb58069'
CALLS=['CALL_001','CALL_035','CALL_041','CALL_209']
FLAGS={'TRANSCRIBE_LANGUAGE_OPTIONS':'en-US,zh-CN','TRANSCRIBE_LANGUAGE_ID_MODE':'multiple','TRANSCRIBE_MAX_SPEAKER_LABELS':'2'}
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8')

def plan():
    from transcribe_eval.collect import new_run_id, local_audio
    baseline=json.loads((ENGINE/'runs'/BASELINE/'metadata.json').read_text())
    originals={c['call_id']:c for c in baseline['calls']}
    run=ENGINE/'runs'/('stt-segmented-'+new_run_id().lower())
    run.mkdir(exist_ok=False)
    records=[]
    for cid in CALLS:
        source=local_audio(ROOT/'evaluation-dataset/datasets'/DATASET,cid,'auto')
        before=digest(source)
        assert before==originals[cid]['audio_sha256']
        chunks=split_pcm(source,run/'raw/segments'/cid)
        assert digest(source)==before
        for c in chunks: c['sha256']=digest(c['path'])
        records.append(dict(call_id=cid,source=str(source),audio_sha256=before,chunks=chunks))
    seconds=sum(c['duration'] for r in records for c in r['chunks'])
    count=sum(len(r['chunks']) for r in records)
    assert count<=32 and seconds<=20*60
    p=dict(run_id=run.name,dataset_id=DATASET,baseline_run=BASELINE,created_at=datetime.now(timezone.utc).isoformat(),
        flags=FLAGS,calls=records,segment_count=count,segment_audio_minutes=seconds/60,
        module_sha256=digest(Path(__file__).with_name('segmentation.py')),
        handler_sha256=digest(ROOT/'lambda/transcribe-trigger/app/handler.py'),
        policy='45s target; quietest 100ms window within 40-50s; 1s context each side; last core >=30s. Exact PCM frame copies. No vocabulary. Timestamp midpoint ownership. Full-call model diarization projected by >=50% word overlap; otherwise abstain.',
        gate='Lower pooled WER errors; entity errors must not increase; speaker accuracy drop <=0.2 percentage points. This is a mixed-language diagnostic, not a full multilingual promotion.')
    save(run/'segment_plan.json',p)
    print(json.dumps(dict(run_id=run.name,segments=count,audio_minutes=seconds/60),indent=2),flush=True)

def execute(runid):
    import boto3
    from transcribe_eval.handler_loader import load_handler
    from transcribe_eval.reference import load_terms
    from transcribe_eval.score import score_run
    if Path(runid).name!=runid: raise ValueError('Run ID only')
    run=ENGINE/'runs'/runid
    p=json.loads((run/'segment_plan.json').read_text())
    if (run/'execution.json').exists(): raise RuntimeError('Execution already attempted; inspect it, never blindly retry')
    assert p['module_sha256']==digest(Path(__file__).with_name('segmentation.py'))
    assert p['handler_sha256']==digest(ROOT/'lambda/transcribe-trigger/app/handler.py')
    for r in p['calls']:
        assert digest(r['source'])==r['audio_sha256']
        for c in r['chunks']: assert digest(c['path'])==c['sha256']
    session=boto3.Session(profile_name='fyp-stt-eval',region_name='ap-southeast-1')
    account=session.client('sts').get_caller_identity()['Account']
    s3=session.client('s3'); bucket='fyp-transcribe-eval-team'
    s3.head_bucket(Bucket=bucket,ExpectedBucketOwner=account)
    notifications=s3.get_bucket_notification_configuration(Bucket=bucket,ExpectedBucketOwner=account)
    assert not any(k!='ResponseMetadata' for k in notifications)
    for k in list(os.environ):
        if k.startswith('TRANSCRIBE_'): del os.environ[k]
    os.environ.update(FLAGS,AWS_PROFILE='fyp-stt-eval',AWS_DEFAULT_REGION='ap-southeast-1',TRANSCRIBE_OUTPUT_BUCKET=bucket)
    handler=load_handler(ROOT/'lambda/transcribe-trigger/app/handler.py')
    state=dict(status='running',jobs=[])
    save(run/'execution.json',state)
    try:
        for r in p['calls']:
            for c in r['chunks']:
                key=f'uploads/{runid}/{r["call_id"]}/segment-{c["index"]:03d}.wav'
                s3.put_object(Bucket=bucket,ExpectedBucketOwner=account,Key=key,Body=Path(c['path']).read_bytes(),ContentType='audio/wav',IfNoneMatch='*')
                jobname=handler._job_name_for_key(key)
                entry=dict(call_id=r['call_id'],index=c['index'],key=key,name=jobname,status='START_REQUESTED')
                state['jobs'].append(entry); save(run/'execution.json',state)
                handler._start_job(bucket,key)
                entry['status']='IN_PROGRESS'; save(run/'execution.json',state)
        print('Started',len(state['jobs']),'segment jobs',flush=True)
        deadline=time.monotonic()+1200
        while any(j['status']!='COMPLETED' for j in state['jobs']):
            if time.monotonic()>deadline: raise TimeoutError('Inspect existing jobs; do not resubmit')
            time.sleep(7)
            for j in state['jobs']:
                if j['status']=='COMPLETED': continue
                job=handler.transcribe.get_transcription_job(TranscriptionJobName=j['name'])['TranscriptionJob']
                assert job.get('IdentifyMultipleLanguages') is True
                assert set(job['LanguageOptions'])=={'en-US','zh-CN'}
                assert not job.get('LanguageIdSettings')
                assert job['Settings']['ShowSpeakerLabels'] and job['Settings']['MaxSpeakerLabels']==2
                j['status']=job['TranscriptionJobStatus']
                if j['status']=='FAILED': raise RuntimeError(job.get('FailureReason','AWS failure'))
                if j['status']=='COMPLETED':
                    target=run/'raw/segment-transcripts'/j['call_id']/f'{j["index"]:03d}.json'
                    target.parent.mkdir(parents=True,exist_ok=True)
                    payload=s3.get_object(Bucket=bucket,Key=f'transcripts/{j["name"]}.json')['Body'].read()
                    json.loads(payload); target.write_bytes(payload)
                    j['transcript_file']=str(target.relative_to(run))
                    print(j['call_id'],j['index'],'COMPLETED',flush=True)
            save(run/'execution.json',state)
        metadata=dict(run_id=runid,dataset_id=DATASET,status='completed',calls=[],handler_environment=FLAGS,
                      pipeline='Full-call Transcribe diarization plus segmented lexical transcription',baseline_run=BASELINE)
        (run/'raw/transcripts').mkdir(exist_ok=True)
        for r in p['calls']:
            full=json.loads((ENGINE/'runs'/BASELINE/'raw/transcripts'/f'{r["call_id"]}.json').read_text(encoding='utf-8'))
            docs=[json.loads((run/'raw/segment-transcripts'/r['call_id']/f'{c["index"]:03d}.json').read_text(encoding='utf-8')) for c in r['chunks']]
            result=assemble(r['chunks'],docs,full)
            relative=f'raw/transcripts/{r["call_id"]}.json'
            save(run/relative,result)
            assert digest(r['source'])==r['audio_sha256']
            metadata['calls'].append(dict(call_id=r['call_id'],status='COMPLETED',transcript_file=relative,audio_sha256=r['audio_sha256']))
        save(run/'metadata.json',metadata)
        reviewed,terms=load_terms(ENGINE/'domain_terms.json')
        summary=score_run(run,terms=terms,terms_reviewed=reviewed)
        state['status']='completed'; save(run/'execution.json',state)
        for j in state['jobs']: s3.delete_object(Bucket=bucket,ExpectedBucketOwner=account,Key=j['key'])
        print(json.dumps({k:summary[k] for k in ['run_id','calls_scored','normalised_wer','entity_wer','speaker_attribution']}),flush=True)
    except Exception as exc:
        state['status']='failed'; state['error']=str(exc); save(run/'execution.json',state)
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--execute'); args=parser.parse_args()
    execute(args.execute) if args.execute else plan()
