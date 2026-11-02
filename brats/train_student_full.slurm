#!/bin/bash
#SBATCH --job-name=brats_full_train
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --exclude=node07
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=train_full_%j.log
# Generic launcher for the fixed-split full-scale BraTS student runs.
# Pass SCRIPT / KD_WEIGHT / SEED via --export, e.g.:
#   sbatch --export=ALL,SCRIPT=train_student_brats_full.py,KD_WEIGHT=0.0,SEED=1 train_student_full.slurm
set -e
echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
echo "SCRIPT=$SCRIPT  KD_WEIGHT=$KD_WEIGHT  SEED=$SEED"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KDProj
cd ~/CoPa-Distill/SegMamba-V2/brats23
python "$SCRIPT"
echo "Done — $(date)"
