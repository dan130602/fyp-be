# Historical v8 development result

Frozen predictions were evaluated on 175 development calls with unchanged evaluator/normalizer1.4.0. The v8 label transfer added924correct assignments versus v6:142calls improved and0worsened. Every call passed exact WER, entityWER and eligible-count invariance. No words, timestamps or model weights changed.

| Pipeline | WER | Entity WER | Speaker attribution | Correct / eligible |
|---|---:|---:|---:|---:|
| Previous Qwen v6 | 2.67% | 5.77% | 98.80% | 201456 / 203907 |
| Refined Qwen v8 | 2.67% | 5.77% | 99.25% | 202380 / 203907 |
| Cached AWS baseline | 3.63% | 5.66% | 99.13% | 159857 / 161266 |

| Language | Calls | Refined Qwen WER | Entity WER | Speaker attribution | AWS attribution |
|---|---:|---:|---:|---:|---:|
| English | 52 | 2.38% | 4.76% | 99.66% | 98.91% |
| Mandarin | 53 | 2.14% | 5.63% | 98.86% | 99.06% |
| Mixed | 35 | 4.38% | 7.22% | 99.21% | 99.25% |
| Singlish | 35 | 2.20% | 5.98% | 99.52% | 99.44% |

The eligible populations differ between ASR systems, so attribution percentages are not DER or a common-word comparison. AWS is ahead in overall entityWER and in Mandarin/Mixed attribution. These repeatedly examined development calls are not held-out evidence. Model inference, alignment and auxiliary turn predictions were reused for the label-only benchmark; fresh service deployment needs those stages and the extra auxiliary ASR pass. Do not compare later fresh inference to this historical cache as if preprocessing/runtime were automatically identical. A replay test of this package must match the saved historical output documents exactly before claiming it is the same refinement.

No raw call output, private audio, per-call score dump, credentials or reference text is stored in this branch. No Supabase publishing or deployment is performed.
