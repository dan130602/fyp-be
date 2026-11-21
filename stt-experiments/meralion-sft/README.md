# Fine-tuning MERaLiON-3-3B-ASR: decoder LoRA + encoder LoRA + full speech adapter

The code that trains, exports and evaluates the fine-tuned speech-to-text model used in ClearCall. The model is run **J5XE, epoch 2**: three components of
[`MERaLiON/MERaLiON-3-3B-ASR`](https://huggingface.co/MERaLiON/MERaLiON-3-3B-ASR) (revision `9f226d738dee62c60a2f43b0ab7b9a2585a132c3`) are trained together and everything else stays frozen.

**Just want to use the model?** You do not need anything in this folder: the weights, model card and results are on Hugging Face,
**[tanedric/meralion-3-3b-asr-clearcall-j5xe](https://huggingface.co/tanedric/meralion-3-3b-asr-clearcall-j5xe)**, and the worker in [`stt-service/meralion`](../../stt-service/meralion/README.md)
runs them (`python stt-service/meralion/tools/get_finetuned_model.py` downloads and verifies the model in one command). This folder is for merging, retraining and auditing.

| Component | How it is trained | Trainable parameters |
|---|---|---:|
| Text decoder (Gemma-2, 26 layers) | LoRA, rank 8, alpha 16, dropout 0.05 on q/k/v/o + gate/up/down (182 modules) | 10,383,360 |
| Speech encoder (Whisper-style, 32 layers) | LoRA, rank 8, alpha 16, dropout 0.05 on q/k/v/out_proj/fc1/fc2 (192 modules) | 5,898,240 |
| Speech adapter (5:1 temporal compression, 6400 -> 2304) | full training (FP32 master weights, BF16 compute) | 55,714,304 |
| **Total** | | **71,995,904 (2.17% of the model)** |

Recipe: plain cross-entropy, uniform sampling with K = 1,067 extra draws per epoch, AdamW (weight decay 0.01), peak LR 2.5e-5 with 5% warm-up, micro-batch 1 with gradient accumulation 8,
gradient clip 1.0, seed 42, 3-epoch schedule; **epoch 2 is the declared checkpoint**. Training data: 140 development calls (2,277 clips, 13.4 h); 35 development calls are held back for
validation. Training used one 24 GB Blackwell MIG slice, about 1.6 GPU-hours.

## Results (held-out 75 calls, run once; normaliser 1.4.0)

| System | Noisy / channel-degraded WER | entity WER | Clean WER | entity WER |
|---|---:|---:|---:|---:|
| Amazon Transcribe | 3.57% | 6.24% | 1.95% | 4.31% |
| MERaLiON, optimised long-audio pipeline, no fine-tuning | 1.80% | 4.01% | 1.01% | 3.59% |
| + decoder LoRA only | 1.34% | 2.29% | 0.59% | 2.06% |
| **+ decoder LoRA + encoder LoRA + full adapter (this recipe)** | **1.18%** | **2.37%** | **0.47%** | **1.68%** |

Known weaknesses (more in the [model card](https://huggingface.co/tanedric/meralion-3-3b-asr-clearcall-j5xe)): the fine-tuned model writes some English comma-grouped amounts and quantities with
dropped or altered digits (`2,000 -> 2,00`), so amounts and quantities are *worse* than with the un-fine-tuned model even though overall WER is better; the audio is synthetic TTS of scripted
dialogues; the product catalogue is closed (8 products); the critical-insertion review of the checkpoint has not been signed off. It is a research checkpoint.

## 1. Merge the trained weights yourself (optional: the Hugging Face repository already has the merged model)

The Hugging Face repository has the trained parameters alone (`lora/J5XE-epoch-2.safetensors`, 288 MB, SHA-256 `8ef570235e9a3845c01b43932b10bc04bb36d086f29913b76c928cc5767369c5`). The file
describes itself (scope `decoder+adapter+encoder`, rank 8, alpha 16, 378 targets, base revision), so nothing else is needed besides the base model and `sft_export.py`.

Needs Python with `torch` and `safetensors`, about 14 GB of free disk (6.6 GB base model + 0.3 GB checkpoint + 6.6 GB merged output) and several GB of RAM (the exporter streams one weight shard
at a time; peak memory was not measured). CPU only, about 4 minutes. Run from the repository root (Linux/macOS shell; in PowerShell set `$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1` first):

```bash
pip install -U huggingface_hub torch safetensors
hf download tanedric/meralion-3-3b-asr-clearcall-j5xe --include "lora/*" --local-dir ./ft
hf download MERaLiON/MERaLiON-3-3B-ASR --revision 9f226d738dee62c60a2f43b0ab7b9a2585a132c3 --local-dir ./base-model

cd stt-experiments/meralion-sft
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python sft_export.py \
  --base ../../base-model --lora ../../ft/lora/J5XE-epoch-2.safetensors \
  --out ../../exports/my-j5xe-merge --id my-j5xe-merge \
  --base-manifest ../../stt-service/meralion/configs/models/meralion-3-3b-asr-base-9f226d7.json \
  --manifest-out ../../stt-service/meralion/configs/models/my-j5xe-merge.json \
  --runtime-config-in ../../stt-service/meralion/configs/runtime/fg175-m2ax-seq-ap-lmonly-m5-fee2-amt1-nobudget-vllm-rp105-sched0.json \
  --runtime-config-out ../../stt-service/meralion/configs/runtime/my-j5xe-merge.json
```

The worker reads the model manifest from `stt-service/meralion/configs/models/<id>.json`, which is why `--manifest-out` points there; `--id my-j5xe-merge` keeps it from overwriting the committed
manifest of the published model. Run the result with `--model-dir exports/my-j5xe-merge --config configs/runtime/my-j5xe-merge.json`.

**Does your merge equal the published one?** Compare the per-file SHA-256 values in your manifest with `release/J5XE-epoch-2.release.json` (`reproducibility.merged_model_files_sha256`). In our test, a
single-threaded merge on an x86-64 Windows laptop CPU (PyTorch 2.15, CPU build) reproduced all 14 files of the model that produced the reported scores; the default multi-threaded merge on the same
machine reproduced 12 of 14 (the two weight shards differ; the float32 sums are probably added in a different order). Other CPUs or PyTorch builds were not tested and may differ in the last bit of
some weights. Such a merge is still a valid model, but the scores in the model card were measured with the published files, so use the published `model/` if you need those numbers.

## 2. Train it again (to reproduce the checkpoint)

You need what git does not contain: the base model, the 175 development recordings (shared privately, about 2.3 GB) and the SFT data package. The package holds reference transcripts, so it is
rebuilt rather than committed; the hashes your rebuilt package must reproduce are in `data-package-v2.sha256.json`.

```bash
python -X utf8 prepare_sft_data.py --tokenizer <base model dir> --out <data dir>            # clips, split, manifest (CPU)
python -u run_align_long.py --inputs long_turn_inputs.json --batch <audio batch dir> \
        --model <Qwen3-ForcedAligner-0.6B dir> --out alignments.json                         # recovers the 92 turns longer than 30 s by forced alignment of the known text
# then train (single GPU; the Slurm wrapper phases training and evaluation so the trainer and vLLM are never resident together)
bash submit_sft.sh train_job.sbatch blackwell 02:15:00 RUN_ID=J5XE OBJECTIVE=ce SAMPLING=uniform K=1067 \
     SCOPE=decoder+adapter+encoder LR=2.5e-5 BASELINE=<E00 evaluation dir> EPOCH0_REFERENCE=<epoch_summary.jsonl of the decoder-only baseline run>
```

The training command is reconstructed from the saved run configuration (Slurm job 369289); `train_job.sbatch` documents every option. The Slurm scripts and cluster paths are those of the SMU Origami
cluster; replace `YOUR_USERNAME` in them with your account and adapt the rest. Tested versions: torch 2.8.0+cu128, transformers 4.57.6, peft 0.17.1, safetensors 0.8.0, meralion-3-asr 0.1.0.

## 3. Files in this folder

| File | Purpose |
|---|---|
| `sft_core.py` | LoRA injection, the full-adapter wrapper (FP32 masters), training scopes, losses and checks |
| `sft_train.py` | the trainer (`--scope decoder+adapter+encoder`) |
| `sft_data.py`, `prepare_sft_data.py`, `run_align_long.py`, `sft_align.py` | data package, sampling plan, forced alignment of long turns |
| `sft_features.py` | optional cache of the audio features (bit-identical to recomputing them) |
| `sft_augment.py` | imported by the trainer; training-time noise augmentation, **off by default and not used for the published model** |
| `sft_export.py` | merges a checkpoint into the base weights; writes the manifest and the runtime configuration |
| `sft_scope_check.py`, `sft_smoke.py`, `stamp_sft_predictions.py` | pre-flight checks of a training scope, a tiny end-to-end smoke test, labelling predictions as fine-tuned |
| `sft_eval.py`, `to_predictions.py`, `sft_select.py`, `sft_report.py`, `pack_eval.py` | scoring through the unchanged worker and the scorer, checkpoint selection tables, figures |
| `train_job.sbatch`, `submit_sft.sh`, `eval_ckpt.sh`, `heldout_ckpt_job.sbatch`, `align_job.sbatch` | Slurm wrappers used on the cluster |
| `data-package-v2.sha256.json` | hashes the rebuilt data package must match |
| `release/J5XE-epoch-2.release.json` | checksums, provenance and results of the published checkpoint |
| `test_*.py`, `test_data/` | tests: `pip install torch transformers safetensors numpy pytest`, then `python -m pytest -q` here (CPU only; three tests skip without the base model's tokenizer or a local checkpoint) |

Other experiments run with the same code (decoder-only LoRA, weighted loss, 3:1 adapter compression, noise augmentation, other scopes) are not part of this folder; the model card compares the model with
the decoder-only LoRA and reports the augmentation variant.

## 4. Licence

The development of this work was assisted by MERaLiON, an AI model developed by the Agency for Science, Technology and Research (A\*STAR), Singapore. The base model's licence (MERaLiON Public Licence,
which includes the Gemma terms and its Prohibited Use Policy, https://ai.google.dev/gemma/prohibited_use_policy) applies to the checkpoint and to anything derived from it. See the Hugging Face
repository for the notices.
