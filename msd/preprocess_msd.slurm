#!/bin/bash
#SBATCH --job-name=msd_prep
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=msd_preprocess_%j.log

# Stage + preprocess an MSD task. Submit with:
#   sbatch --export=ALL,TASK=spleen preprocess_msd.slurm
# NOTE: preprocessing is CPU-only; --gres=gpu:1 only guarantees scheduling on the
# GPU partition (same note as amos preprocess). Switch to a CPU partition if available.

set -e
TASK=${TASK:-spleen}
echo "Job $SLURM_JOB_ID on $(hostname) — $(date)  TASK=${TASK}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate KDProj
PROJ=${PROJ:-$HOME/SingleTeacherKD}
cd $PROJ/msd_${TASK}
export TASK
python stage_msd.py
python 2_preprocessing_msd.py
echo "Done — $(date). Preprocessed cases in ./data/fullres/train/"
ls -1 ./data/fullres/train/*.npz | wc -l
