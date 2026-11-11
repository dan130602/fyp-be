> Historical experiment notes. See `stt-experiments/meralion/README.md` and its audited results for the current unchanged-weight recommendation and corrected model comparisons.

# Segmented STT experiment: prepared, not yet executed

Status: executed successfully after login refresh; all 19 jobs completed. Candidate failed the accuracy gate. See RESULTS.md. The preparation and resume instructions below are retained for audit; do not rerun this completed execution.

## Frozen plan

- Run: stt-segmented-20261108t053907z-b3f0bb6f
- Calls: CALL_001, CALL_035, CALL_041, CALL_209
- Segments: 19; additional audio submitted on execution: 15.57 minutes.
- Reuses existing full-call baseline output for speaker diarization and comparison. A deployed two-pass pipeline would also incur the full-call transcription cost.
- Core chunks target 45 seconds, with quiet boundaries selected within 40–50 seconds and one second of context on each side. The last core is at least 30 seconds. Clip duration does not guarantee 30 seconds of speech.
- Raw PCM frame bytes, encoding, sample rate and channel count are preserved. Original WAV files are never overwritten. Temporary clips are under the git-ignored run directory.
- English–Mandarin multiple-language identification remains enabled; no vocabulary or fixed-language routing. No expected answers or benchmark language labels are used by the segmentation/reassembly module.
- Timestamp midpoint ownership assigns overlap words to one core. All timestamps are shifted back to full-call time. No text rewriting or selection using reference answers.
- Segment-local speaker IDs are discarded. Existing full-call Transcribe speaker evidence is projected by time overlap; words with less than 50% overlap remain unassigned. This limitation must be reported alongside measured speaker accuracy and coverage.

## Checks

- Four offline tests passed: exact sample preservation and complete core coverage, overlap ownership/global timing, missing-segment failure and speaker abstention, punctuation and real repetition.
- Live source hashes match the baseline. Every temporary clip has been checked against the exact source PCM byte slice.
- Candidate must reduce pooled WER errors without increasing entity errors; speaker accuracy may not drop by more than 0.2 percentage points. This mixed-language trial alone cannot approve production deployment.

## Resume

Run the following once in PowerShell and complete the browser sign-in:

```powershell
& "$env:LOCALAPPDATA\Programs\Amazon\AWSCLIV2ws.exe" login --profile fyp-stt-eval --region ap-southeast-1
```

After authentication, the prepared runner accepts `--execute stt-segmented-20261108t053907z-b3f0bb6f`. It refuses to resubmit an execution already attempted. Inspect its execution record on failure; do not retry blindly.

## Implementation

- `stt-experiments/segmented-v1/segmentation.py`: audio-only segment selection and assembly.
- `stt-experiments/segmented-v1/run_trial.py`: offline planning and bounded AWS execution through the existing production handler; unchanged scorer.
- `stt-experiments/segmented-v1/test_segmentation.py`: offline checks.

No production Lambda change, scorer/dataset edit, held-out use, deployment, publishing, commit or push.

Reference: [AWS batch language identification](https://docs.aws.amazon.com/transcribe/latest/dg/lang-id-batch.html) recommends at least 30 seconds of speech for best results.
