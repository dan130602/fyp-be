# Refined Qwen speaker pipeline, without denoising

This folder packages the original-audio MERaLiON + KMeans Pyannote Community-1 + Qwen3 ForcedAligner path and the frozen v3 -> v6 -> v8 speaker-label refinement. The seven validated label modules preserve the historical source, with UTF-8 encoding and LF line endings normalized; `validated-source-hashes.json` records them. Only private batch/CLI plumbing was made portable. No denoising model, denoising driver, training adapter, reference transcript or raw evaluation output is included.

The historical 175-call development result is 99.25% speaker attribution, 2.67% WER and 5.77% entity WER. See RESULTS.md for denominators, language breakdown and limitations. These are saved-prediction results, not a promise that fresh GPU inference reproduces the scores. The 146 environment recordings plus 29 clean originals had different historical preprocessing provenance. A fresh comparison must run its original control under the same runtime/resampling/decoding settings; historical cached ASR alone is not a matched denoising control.

Use `pipeline.refine_document(document, exclusive, regular, auxiliary_items)` for the complete speaker-only chain. Inputs must be predicted Qwen `qwen_refined` items, original-audio speaker timelines and the predicted auxiliary speaker-turn transcript from the same recording. The chain preserves text, timestamps, item order, input objects and the unchanged normalizer's hypothesis-word/source-origin signature. The contract guard imports the existing `evaluation-engine/transcribe_eval` utilities without changing them. It never reads reference text or scores. Normalizer 1.4.0 is required and checked.

For a single saved call, from the repository root:

```bash
python -m pip install -r stt-experiments/meralion/qwen-speakers/requirements-labels.txt
python stt-experiments/meralion/qwen-speakers/pipeline.py --transcript /private/aligned/CALL_001.json --speakers /private/speakers/CALL_001.json --auxiliary /private/turn_first/CALL_001.json --output /private/refined/CALL_001.json
```

The output must be new; existing files are never overwritten. `pipeline.py` accepts either a Transcribe-shaped document or an aligned prediction wrapper containing `qwen_refined`. For production, require a common source-audio hash for every input and make the same unchanged normalizer utility available. The standalone wrapper enforces hashes when the aligned wrapper provides one. No Lambda handler selects this pipeline automatically.

For fresh GPU stage commands, environments, private manifests and teammate setup, see ../QWEN_SPEAKER_HANDOVER.md. Fresh deployment needs the extra auxiliary speaker-turn ASR pass; this is not just Pyannote plus Qwen. Keep AWS as the current production default until the owner integrates and validates the GPU service. No infrastructure, frontend, NLP, API contract or model weights are changed here.
