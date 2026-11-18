"""Hypothesis-only text/normalization contract; never receives gold or scores."""
import unicodedata

def signature(items):
 from transcribe_eval.hypothesis import raw_words
 from transcribe_eval.normalize import normalize_words_traced
 rows=[];breaks=set();commas=set()
 for z in items:
  if z['type']=='punctuation':
   mark=unicodedata.normalize('NFKC',z['alternatives'][0]['content']).replace('\u3002','.').replace('\u3001',',')
   if rows and mark in set('.?!;:,'):
    index=len(rows)-1;breaks.add(index)
    if mark==',':commas.add(index)
    else:commas.discard(index)
   continue
  rows.append(dict(item=len(rows),content=z['alternatives'][0]['content'],speaker=z.get('speaker_label')))
 words,owners,phrase,commas,bounds=raw_words(rows,breaks,commas)
 tokens,origins,_,_=normalize_words_traced(words,phrase,boundaries_after=bounds,commas_after=commas)
 return tuple((token,tuple(sorted({owners[i] for i in origin}))) for token,origin in zip(tokens,origins))

def equivalent(before,after):return signature(before)==signature(after)
