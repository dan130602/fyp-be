# Audited comparison of saved STT models

Eight identical development recordings, two per language class. Scores use Edric's unchanged CLI and normalizer 1.4.0. Hosted text outputs were rebuilt using the corrected numeric-symbol-preserving adapter; native AWS items and audited Deepgram items retain provider metadata. Qwen's sentence-derived speaker labels were preserved by exact character alignment. No provider API requests or new model inference were performed.

| Group | Model/configuration | WER | Entity WER | Speaker attribution |
|---|---|---:|---:|---:|
| All eight | Amazon Transcribe | 2.52% (156/6185) | 2.82% (27/956) | 99.66% (4382/4397) |
| All eight | Local MERaLiON energy + fee | 2.44% (151/6185) | 3.24% (31/956) | Unavailable |
| All eight | Hosted MERaLiON API | 0.91% (56/6185) | 1.88% (18/956) | Unavailable |
| All eight | Fun-ASR Flash | 1.10% (68/6185) | 2.30% (22/956) | Unavailable |
| All eight | Qwen hosted | 1.46% (90/6185) | 3.56% (34/956) | 99.66% (2898/2908) |
| All eight | Deepgram Nova-3 | 12.50% (773/6185) | 7.95% (76/956) | 95.16% (4204/4418) |
| All eight | Whisper small | 8.88% (549/6185) | 10.46% (100/956) | Unavailable |
| All eight | Transcribe + 11-term vocabulary | 2.54% (157/6185) | 2.82% (27/956) | 99.66% (4383/4398) |
| English | Amazon Transcribe | 0.95% (11/1156) | 1.64% (4/244) | 99.25% (1063/1071) |
| English | Local MERaLiON energy + fee | 1.73% (20/1156) | 0.41% (1/244) | Unavailable |
| English | Hosted MERaLiON API | 0.78% (9/1156) | 2.46% (6/244) | Unavailable |
| English | Fun-ASR Flash | 1.30% (15/1156) | 2.05% (5/244) | Unavailable |
| English | Qwen hosted | 1.04% (12/1156) | 3.69% (9/244) | 99.07% (1067/1077) |
| English | Deepgram Nova-3 | 1.73% (20/1156) | 2.05% (5/244) | 99.82% (1117/1119) |
| English | Whisper small | 0.87% (10/1156) | 2.05% (5/244) | Unavailable |
| English | Transcribe + 11-term vocabulary | 0.95% (11/1156) | 1.64% (4/244) | 99.25% (1063/1071) |
| Mandarin | Amazon Transcribe | 1.88% (38/2017) | 2.47% (8/324) | 99.73% (1111/1114) |
| Mandarin | Local MERaLiON energy + fee | 2.08% (42/2017) | 7.10% (23/324) | Unavailable |
| Mandarin | Hosted MERaLiON API | 0.94% (19/2017) | 2.47% (8/324) | Unavailable |
| Mandarin | Fun-ASR Flash | 0.45% (9/2017) | 0.93% (3/324) | Unavailable |
| Mandarin | Qwen hosted | 0.35% (7/2017) | 0.31% (1/324) | 100.00% (196/196) |
| Mandarin | Deepgram Nova-3 | 27.07% (546/2017) | 17.28% (56/324) | 99.57% (934/938) |
| Mandarin | Whisper small | 3.37% (68/2017) | 8.95% (29/324) | Unavailable |
| Mandarin | Transcribe + 11-term vocabulary | 1.88% (38/2017) | 2.47% (8/324) | 99.73% (1111/1114) |
| Mixed English–Mandarin | Amazon Transcribe | 5.32% (90/1692) | 4.95% (10/202) | 99.70% (1003/1006) |
| Mixed English–Mandarin | Local MERaLiON energy + fee | 4.37% (74/1692) | 2.48% (5/202) | Unavailable |
| Mixed English–Mandarin | Hosted MERaLiON API | 1.30% (22/1692) | 1.98% (4/202) | Unavailable |
| Mixed English–Mandarin | Fun-ASR Flash | 0.47% (8/1692) | 0.99% (2/202) | Unavailable |
| Mixed English–Mandarin | Qwen hosted | 2.78% (47/1692) | 5.94% (12/202) | 100.00% (413/413) |
| Mixed English–Mandarin | Deepgram Nova-3 | 10.52% (178/1692) | 6.93% (14/202) | 81.30% (900/1107) |
| Mixed English–Mandarin | Whisper small | 25.53% (432/1692) | 29.21% (59/202) | Unavailable |
| Mixed English–Mandarin | Transcribe + 11-term vocabulary | 5.26% (89/1692) | 4.95% (10/202) | 99.70% (1005/1008) |
| Singlish | Amazon Transcribe | 1.29% (17/1320) | 2.69% (5/186) | 99.92% (1205/1206) |
| Singlish | Local MERaLiON energy + fee | 1.14% (15/1320) | 1.08% (2/186) | Unavailable |
| Singlish | Hosted MERaLiON API | 0.45% (6/1320) | 0.00% (0/186) | Unavailable |
| Singlish | Fun-ASR Flash | 2.73% (36/1320) | 6.45% (12/186) | Unavailable |
| Singlish | Qwen hosted | 1.82% (24/1320) | 6.45% (12/186) | 100.00% (1222/1222) |
| Singlish | Deepgram Nova-3 | 2.20% (29/1320) | 0.54% (1/186) | 99.92% (1253/1254) |
| Singlish | Whisper small | 2.95% (39/1320) | 3.76% (7/186) | Unavailable |
| Singlish | Transcribe + 11-term vocabulary | 1.44% (19/1320) | 2.69% (5/186) | 99.92% (1204/1205) |

The latest original-energy rerun matched every saved audited energy transcript on all 175 calls. Local MERaLiON here uses the fee-only formatting condition; the other models use their provider transcripts without that additional cleanup.

These are pilot results, not full-dataset rankings. The exact hosted MERaLiON checkpoint has not been verified as the same as the open-weight local model. Whisper uses known evaluation language classes to select language settings, so this does not measure automatic routing. Deepgram's Mandarin errors include Simplified/Traditional character differences retained by the scorer. Speaker attribution denominators vary by provider and Qwen labels were adapted at sentence level.

Full 175-call evidence: published AWS dashboard 1.63% WER / 3.12% entity WER; local MERaLiON original energy 1.74% / 4.35%, or 1.72% / 4.15% with fee formatting. AWS full-set native output was not independently rescored here. The owner-held-out set remains unused. The earlier LoRA experiment changed adapter weights and is excluded from the current unchanged-weight comparison.
