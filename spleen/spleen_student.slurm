#!/bin/bash
#SBATCH --job-name=spl_student
#SBATCH --partition=gpu-l40s
#SBATCH --exclude=node07
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=spleen_%x_%j.log
# Submit FROM ~/CoPa-Distill/SegMamba-V2/spleen/ . Separators: this launcher takes plain values only
# (no commas inside a value), because sbatch --export splits on commas.
#
#  1) sanity check of the teacher inside this pipeline (expect ~0.967 on val, as in its own log):
#       sbatch --job-name=spl_chk --time=02:00:00 --export=ALL,MODE=val,ARCH=teacher spleen_student.slurm
#  2) smoke test (a few minutes):
#       sbatch --job-name=spl_smoke --time=01:00:00 --export=ALL,KD_MODE=dist,SEED=1,EPOCHS=1,STEPS=5,VAL_EVERY=1,TAG=_smoke spleen_student.slurm
#  3) real runs, 3 seeds each:
#       sbatch --job-name=spl_none --array=1-3 --export=ALL,KD_MODE=none spleen_student.slurm
#       sbatch --job-name=spl_dist --array=1-3 --export=ALL,KD_MODE=dist spleen_student.slurm
#  4) test, once per finished run (needs a GPU; ~minutes):
#       sbatch --job-name=spl_test --time=02:00:00 --export=ALL,MODE=test,KD_MODE=none,SEED=1 spleen_student.slurm
set -eo pipefail
echo "Job $SLURM_JOB_ID on $(hostname) - $(date)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segmamba
cd ~/CoPa-Distill/SegMamba-V2/spleen
export SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-1}}
echo "MODE=${MODE:-train} ARCH=${ARCH:-student} KD_MODE=${KD_MODE:-none} SEED=$SEED EPOCHS=${EPOCHS:-250} TAG=${TAG:-}"
nvidia-smi -L
python spleen_student.py
echo "Done - $(date)"
