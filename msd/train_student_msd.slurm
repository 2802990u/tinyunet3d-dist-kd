#!/bin/bash
#SBATCH --job-name=msd_student
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=msd_student_%j.log

# TinyUNet3D student, baseline vs KD. Submit e.g.:
#   sbatch --export=ALL,TASK=spleen,KD_MODE=none,SEED=1 train_student_msd.slurm   # baseline
#   sbatch --export=ALL,TASK=spleen,KD_MODE=kl,SEED=1   train_student_msd.slurm   # KD
# Paper runs: KD_MODE in {none,kl} x SEED in {1,2,3} per task (dist optional).
set -e
TASK=${TASK:-spleen}
echo "Job $SLURM_JOB_ID on $(hostname) — $(date)  TASK=${TASK} KD_MODE=${KD_MODE} SEED=${SEED}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KDProj
PROJ=${PROJ:-$HOME/SingleTeacherKD}
cd $PROJ/msd_${TASK}
export TASK
python train_student_msd.py
echo "Done — $(date)."
