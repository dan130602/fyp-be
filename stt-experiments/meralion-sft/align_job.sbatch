#!/bin/bash -l
# Forced alignment of the 92 long development turns (known reference text vs audio). CPU by default (no GPU allocation); `ALIGN_DEVICE=cuda:0` with a GPU submission is the fallback.
#   sbatch --partition=student --qos=studentqos --account=student --cpus-per-task=4 --mem=32G --time=00:40:00 align_job.sbatch
#SBATCH --job-name=sft-align
#SBATCH --partition=student
#SBATCH --account=student
#SBATCH --qos=studentqos
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.log
set -eo pipefail
module load Python/3.12.8-GCCcore-13.3.0
SFT=/common/scratch/users/e/YOUR_USERNAME/meralion/sft
PILOT=$HOME/meralion-pilot
OUT=$SFT/eval/align-$SLURM_JOB_ID
mkdir -p "$OUT"
export OMP_NUM_THREADS=4
$PILOT/.venv-aligner/bin/python -u $SFT/code/run_align_long.py --inputs $SFT/data-v1/long_turn_inputs.json --batch $PILOT/batch --model $PILOT/models/forced-aligner \
    --out $OUT/alignments.json --device ${ALIGN_DEVICE:-cpu} ${ALIGN_LIMIT:+--limit $ALIGN_LIMIT} 2>&1 | tee $OUT/align.log
echo "results in $OUT"
