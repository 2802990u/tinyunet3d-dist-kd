#!/bin/bash
#SBATCH --job-name=segv2_repro_train
#SBATCH --partition=gpu-l40s
#SBATCH --exclude=node07
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --output=/path/to/home/AGILEMamba/segv2_repro_train_%j.log

# ============================================================================
# STEP 2: train SegMamba-V2 from scratch on OUR 876/125/250 split, with their
# recipe, for a controlled head-to-head against AGILE-Mamba.
#
# WHY: our only direct comparison so far used their released checkpoint on the
# 41 cases held out by both splits (AGILE-Mamba 92.62 vs SegMamba-V2 91.85).
# 41 cases carries +/-1.5-2 points of standard error and the two models were
# trained on different data, so it is suggestive, not conclusive. Training
# their architecture on our data with their recipe removes both objections.
#
# 3 GPUs: their recipe is 4x A100 with batch 2/GPU (effective 8). Our nodes
# have 3x L40S, giving effective 6. Closer than single-GPU's effective 2.
#
# Runs entirely on /mnt/scratch. MedicalDataset unpacks .npz -> .npy on first
# use, which roughly doubles the 16 GB of preprocessed data; the home quota
# (223G of 250G) has no room for that, scratch has 471 TB.
#
# 1000 epochs x 876 cases is long. --time is at the 72h cap, so expect to
# requeue. The trainer writes best_model_*.pt and final_model_*.pt every
# val_every=2 epochs, so a requeue loses at most a couple of epochs -- but
# note their Trainer has no resume-from-checkpoint path, so a restart begins
# from scratch unless we add one. Check progress before the wall clock runs
# out and tell me if we need to add resume support.
# ============================================================================

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segmamba

WORK=/path/to/scratch/segv2_repro
SEGV2_SRC=/path/to/home/AGILEMamba/SegMamba-V2/brats23

# Their code imports light_training and models_segmamba by relative package
# name, so the script has to run from the directory that contains them.
# segv2_repro_prepare.slurm already copied both into $WORK.
for d in light_training models_segmamba; do
    [ -e "$WORK/$d" ] || cp -r "$SEGV2_SRC/$d" "$WORK/"
done
cp /path/to/home/AGILEMamba/segv2_repro_train.py "$WORK/"

cd "$WORK"

echo "=== GPUs visible to this job ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "=== split files ==="
wc -l "$WORK/splits"/*.txt
echo "=== preprocessed cases ==="
ls "$WORK/data/fullres/train"/*.npz | wc -l
echo "=== scratch free ==="
df -h /mnt/scratch | tail -1

echo "=== launching training ==="
python "$WORK/segv2_repro_train.py"

echo "=== done ==="
ls -la "$WORK/logs/segmambav2_ourssplit/model" || true
df -h /mnt/scratch | tail -1
