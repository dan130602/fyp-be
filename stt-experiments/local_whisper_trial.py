"""Offline four-call feasibility trial; no reference text is used in inference."""
import json, sys, time, hashlib, re
from pathlib import Path
from datetime import datetime, timezone
from faster_whisper import WhisperModel

ROOT = Path('C:/Users/user/Documents/fyp-working/fyp-be-latest')
ENGINE = ROOT / 'evaluation-engine'
sys.path.insert(0, str(ENGINE))
DATASET = 'advisory_execution_multilingual_v1_development'
DATA = ROOT / 'evaluation-dataset/datasets' / DATASET
MODEL = Path('C:/Users/user/.cache/huggingface/hub/models--Systran--faster-whisper-base/snapshots/ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66')

def convert(segments):
    items, texts = [], []
    for segment in segments:
        texts.append(segment.text)
        for word in segment.words or []:
            for token in re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", word.word, re.UNICODE):
                item = {'type': 'pronunciation' if any(c.isalnum() for c in token) else 'punctuation',
                        'alternatives': [{'content': token}]}
                if item['type'] == 'pronunciation':
                    item.update(start_time=str(word.start), end_time=str(word.end))
                items.append(item)
    return {'results': {'items': items, 'transcripts': [{'transcript': ''.join(texts)}]}}

def main():
    run_id = 'whisper-base-cpu-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = ENGINE / 'runs' / run_id
    (run / 'raw/transcripts').mkdir(parents=True, exist_ok=False)
    metadata = dict(run_id=run_id, dataset_id=DATASET, status='running', calls=[],
                    model='Systran/faster-whisper-base', model_revision=MODEL.name,
                    parameters=dict(device='cpu', compute_type='int8', beam_size=5, language=None,
                                    word_timestamps=True, vad_filter=False),
                    limitations=['No speaker diarization; speaker metric unavailable', 'Confidence omitted; not comparable to AWS'])
    model = WhisperModel(str(MODEL), device='cpu', compute_type='int8', cpu_threads=4, local_files_only=True)
    for call_id in ['CALL_059','CALL_064','CALL_094','CALL_147']:
        paths = list((DATA / 'calls' / call_id / 'renders').glob('*/audio.*'))
        assert len(paths) == 1, paths
        audio = paths[0]
        print('Starting', call_id, flush=True)
        started = time.monotonic()
        segments, info = model.transcribe(str(audio), beam_size=5, word_timestamps=True, vad_filter=False)
        result = convert(segments)
        relative = 'raw/transcripts/' + call_id + '.json'
        (run / relative).write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
        metadata['calls'].append(dict(call_id=call_id, status='COMPLETED', transcript_file=relative,
            render_id=audio.parent.name, audio_bytes=audio.stat().st_size,
            audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
            manifest_normalized_audio_sha256=hashlib.sha256(audio.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
            detected_language=info.language, audio_seconds=info.duration, inference_seconds=round(time.monotonic()-started,2)))
        (run / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('Completed', call_id, metadata['calls'][-1]['inference_seconds'], flush=True)
    metadata['status'] = 'completed'
    (run / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    from transcribe_eval.reference import load_terms
    from transcribe_eval.score import score_run
    reviewed, terms = load_terms(ENGINE / 'domain_terms.json')
    summary = score_run(run, terms=terms, terms_reviewed=reviewed)
    print(json.dumps(summary), flush=True)
    print('RUN_DIRECTORY', run, flush=True)

if __name__ == '__main__':
    main()
