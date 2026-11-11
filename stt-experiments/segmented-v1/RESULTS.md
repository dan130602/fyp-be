> Historical experiment notes. See `stt-experiments/meralion/README.md` and its audited results for the current unchanged-weight recommendation and corrected model comparisons.

# Segmented STT trial results

Decision: failed the pilot gate. Do not deploy or run all 175 calls with this candidate.

Four mixed-language development calls; the baseline below is the matching four-call subset, not the entire earlier 16-call run.

| Metric | Full-call baseline | Segmented candidate |
|---|---:|---:|
| WER | 5.63% | 6.18% |
| Entity WER | 4.49% | 4.75% |
| Speaker attribution | 99.52% | 99.33% |

Word/character errors: 186 → 204 / 3302.
Entity errors: 17 → 18 / 379.
Substitutions/deletions/insertions: 86/95/5 → 83/117/4.

| Call | Baseline WER | Candidate WER | Baseline entity WER | Candidate entity WER | Baseline speaker | Candidate speaker |
|---|---:|---:|---:|---:|---:|---:|
| CALL_001 | 5.60% | 8.00% | 10.00% | 10.00% | 99.83% | 99.10% |
| CALL_035 | 5.02% | 5.02% | 2.27% | 3.03% | 99.54% | 99.54% |
| CALL_041 | 7.00% | 7.00% | 0.00% | 0.00% | 98.91% | 98.91% |
| CALL_209 | 4.98% | 4.61% | 7.29% | 7.29% | 100.00% | 100.00% |

## Method and limitations

- 19 segment jobs, 15.57 additional audio minutes. Reused the previous full-call Transcribe output for global speaker evidence; a deployed two-pass system would also pay for that first pass.
- Approximately 45-second cores selected at quiet audio boundaries, one second of context each side, timestamp midpoint ownership on reassembly. Exact PCM byte slices; original file hashes verified before and after inference.
- Effective settings: TRANSCRIBE_LANGUAGE_OPTIONS=en-US,zh-CN; TRANSCRIBE_LANGUAGE_ID_MODE=multiple; TRANSCRIBE_MAX_SPEAKER_LABELS=2. No vocabulary or custom language model. Actual job settings checked for every segment.
- Full-call speaker evidence is projected by timestamp overlap, with abstention below 50% overlap. 5/2175 output pronunciation items have no speaker label. Speaker accuracy includes the consequences of abstention; item coverage and eligible denominators can differ between outputs.
- Eligible speaker items: baseline 2092/2102, candidate 2079/2093.
- Existing scorer and normalizer 1.4.0 unchanged. Four offline tests passed before execution. Reference text and benchmark labels were not used in segmentation, language routing or assembly.
- This sample contains only mixed-language calls. English, Mandarin-only and Singlish regression testing has not been performed for this pipeline. No held-out data used.
- No production deployment, persistent settings change, publishing, commit or push. Original audio and datasets unchanged. Temporary S3 input clips were removed after successful scoring; normal output transcripts and local audit files remain.

## Run identifiers

- Baseline subset: `stt-vocab-broader-baseline-20261107t175401z-bfb58069`
- Candidate: `stt-segmented-20261108t053907z-b3f0bb6f`
