#!/bin/bash
#SBATCH --job-name=msd_eval
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=msd_eval_%j.log

# Held-out test eval. Submit e.g.:
#   sbatch --export=ALL,TASK=spleen,ARCH=student,MODEL_PATH='./logs/student_spleen_kl1_s1/model/best_model_*.pt' eval_msd_test.slurm
#   sbatch --export=ALL,TASK=spleen,ARCH=segmamba,MODEL_PATH='./logs/ssm_teacher_spleen/model/best_model_*.pt' eval_msd_test.slurm
set -e
TASK=${TASK:-spleen}
echo "Job $SLURM_JOB_ID on $(hostname) — $(date)  TASK=${TASK} ARCH=${ARCH} MODEL_PATH=${MODEL_PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KDProj
PROJ=${PROJ:-$HOME/SingleTeacherKD}
cd $PROJ/msd_${TASK}
export TASK ARCH MODEL_PATH
python eval_msd_test.py
echo "Done — $(date)."
