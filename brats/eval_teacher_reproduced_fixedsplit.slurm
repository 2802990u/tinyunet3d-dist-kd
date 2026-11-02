#!/bin/bash
#SBATCH --job-name=eval_teacher_repro
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --exclude=node07
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=eval_teacher_repro_%j.log
# Evaluate the REPRODUCED (leak-free) SegMamba-V2 checkpoint on the fixed
# 250-case test split -- this is the real "Teacher (ours)" number.
#   sbatch eval_teacher_reproduced_fixedsplit.slurm
set -e
echo "Job $SLURM_JOB_ID on $(hostname) — $(date)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KDProj
cd ~/CoPa-Distill/SegMamba-V2/brats23
python eval_teacher_reproduced_fixedsplit.py
echo "Done — $(date). teacher_reproduced_fixedsplit_test250_percase.npy now holds the per-case Dice."
