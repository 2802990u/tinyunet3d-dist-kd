#!/bin/bash
#SBATCH --job-name=msd_ssm_t
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=msd_ssm_teacher_%j.log

# SegMamba teacher on an MSD task. Submit with:
#   sbatch --export=ALL,TASK=spleen train_ssm_teacher_msd.slurm
set -e
TASK=${TASK:-spleen}
echo "Job $SLURM_JOB_ID on $(hostname) — $(date)  TASK=${TASK}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KDProj
PROJ=${PROJ:-$HOME/SingleTeacherKD}
cd $PROJ/msd_${TASK}
export TASK
python train_ssm_teacher_msd.py
echo "Done — $(date). Teacher in ./logs/ssm_teacher_${TASK}/model/"
