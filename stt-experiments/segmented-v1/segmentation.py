"""Audio-only PCM segmentation and two-pass ASR assembly. No dataset imports."""
from array import array
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import sys
import wave


def split_pcm(source, destination):
    """Copy PCM frames into unique WAV files; retain original encoding and samples."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with wave.open(str(source), 'rb') as audio:
        params = audio.getparams()
        if params.comptype != 'NONE' or params.sampwidth != 2:
            raise ValueError('Only uncompressed 16-bit PCM WAV is supported; never transcode')
        data = audio.readframes(params.nframes)
    rate, frames, width = params.framerate, params.nframes, params.nchannels * params.sampwidth
    samples = array('h', data)
    if sys.byteorder != 'little':
        samples.byteswap()
    boundaries = [0]
    # Prefer a quiet 100 ms window near 45 s. Never leave a final core shorter than 30 s.
    while frames - boundaries[-1] > 75 * rate:
        start = boundaries[-1]
        candidates = range(start + 40 * rate, start + 50 * rate + 1, max(1, rate // 10))
        half = max(1, rate // 20)
        def energy(frame):
            block = samples[(frame-half)*params.nchannels:(frame+half)*params.nchannels]
            return sum(v*v for v in block) / max(1, len(block))
        cut = min(candidates, key=lambda x: (energy(x), abs(x-(start+45*rate))))
        boundaries.append(cut)
    boundaries.append(frames)
    chunks = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        first, last = max(0, start-rate), min(frames, end+rate)
        payload = data[first*width:last*width]
        path = destination / f'segment-{index:03d}.wav'
        with wave.open(str(path), 'wb') as target:
            target.setparams(params)
            target.writeframes(payload)
        with wave.open(str(path), 'rb') as check:
            assert check.readframes(check.getnframes()) == payload
        chunks.append(dict(index=index, path=str(path), start_frame=first, end_frame=last,
            core_start=start/rate, core_end=end/rate, offset=first/rate,
            duration=(last-first)/rate, final=index == len(boundaries)-2))
    return chunks


def speaker_for(start, end, full_items):
    """Project actual full-call ASR diarization by time overlap; abstain on weak evidence."""
    weights = defaultdict(float)
    for item in full_items:
        if item.get('type') != 'pronunciation' or not item.get('speaker_label'):
            continue
        overlap = max(0, min(end, float(item['end_time']))-max(start, float(item['start_time'])))
        weights[item['speaker_label']] += overlap
    if not weights or end <= start:
        return None
    label = max(weights, key=weights.get)
    return label if weights[label] >= 0.5*(end-start) else None


def assemble(chunks, documents, full_call):
    """Own each timestamp midpoint once; retain segment text, map global times/speakers."""
    if len(chunks) != len(documents):
        raise ValueError('Every segment must have exactly one transcript')
    output = []
    for chunk, doc in zip(chunks, documents):
        kept = False
        for raw in doc['results']['items']:
            if raw['type'] == 'punctuation':
                if kept:
                    output.append(deepcopy(raw))
                continue
            item = deepcopy(raw)
            start, end = float(item['start_time'])+chunk['offset'], float(item['end_time'])+chunk['offset']
            midpoint = (start+end)/2
            kept = chunk['core_start'] <= midpoint < chunk['core_end'] or (chunk['final'] and midpoint == chunk['core_end'])
            if not kept:
                continue
            item['start_time'], item['end_time'] = str(round(start, 6)), str(round(end, 6))
            item.pop('speaker_label', None)  # Per-segment speaker IDs cannot be stitched directly.
            label = speaker_for(start, end, full_call['results']['items'])
            if label:
                item['speaker_label'] = label
            output.append(item)
    text = ''
    for item in output:
        text += ('' if item['type'] == 'punctuation' or not text else ' ') + item['alternatives'][0]['content']
    return {'results': {'items': output, 'transcripts': [{'transcript': text}]}}
