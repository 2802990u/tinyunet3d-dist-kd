#!/bin/bash
#SBATCH --job-name=segv2_repro_eval
#SBATCH --partition=gpu-l40s
#SBATCH --exclude=node07
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/home/AGILEMamba/segv2_repro_eval_%j.log

# STEP 3: score the reproduced SegMamba-V2 on our 250-case test set, in
# WT/TC/ET regions -- the number their paper reports and the number
# AGILE-Mamba is measured with.
#
# The training run (job 1882370) completed all 1000 epochs; its apparent
# "failure" was batchgenerators failing to shut down its worker threads AFTER
# training finished. Best val (class-wise) 0.8921.
#
# Uses best_model_*.pt, selected on our 125-case validation set -- NOT
# final_model (last epoch). This is the first time our 250 test cases are
# shown to this model.

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segmamba

WORK=/path/to/scratch/segv2_repro
cp /path/to/home/AGILEMamba/segv2_repro_eval_test.py "$WORK/"
cd "$WORK"

echo "=== checkpoints available ==="
ls -la "$WORK/logs/segmambav2_ourssplit/model/"

python "$WORK/segv2_repro_eval_test.py"
