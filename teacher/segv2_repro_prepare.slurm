#!/bin/bash
#SBATCH --job-name=segv2_repro_prepare
#SBATCH --partition=gpu-l40s
#SBATCH --exclude=node07
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/home/AGILEMamba/segv2_repro_prepare_%j.log

# ============================================================================
# STEP 1 of the SegMamba-V2 reproduction on OUR split.
#
# Runs THEIR preprocessing pipeline over ALL 1251 BraTS2023-GLI cases, so that
# afterwards we can train their model with their recipe on our 876/125/250
# partition.
#
# WHY SCRATCH: their fullres output is ~66 MB/case (measured on the 41-case
# run: 2.7G / 41). 1251 cases -> ~82 GB, plus their unpack_dataset step which
# decompresses .npz -> .npy on top of that. Home quota has ~27 GB free.
# /mnt/scratch has 472 TB free and is a SEPARATE filesystem, not counted
# against the /users quota.
#
# SAFETY: the rename step rewrites filenames IN PLACE
#   BraTS-GLI-XXXXX-000-t2w.nii.gz  ->  t2w.nii.gz
# which would destroy our own brats_dataset.py's view of the data. We COPY the
# raw tree to scratch first and only ever rename the copy. The original under
# $HOME is never touched. Do not "optimise" this into a symlink.
#
# CHANGES vs the first attempt (job 1881386, failed at 00:00:54):
#   1. The source folder contains a loose file, segmamba_best.pth, alongside
#      the case directories. Their 1_rename script does os.listdir() on every
#      entry and crashed with NotADirectoryError. We now exclude non-directory
#      entries from the copy and delete any that slipped through.
#   2. The rename is done inline and IDEMPOTENTLY (already-short names are
#      skipped) so a re-run or a requeue resumes instead of erroring on the
#      ~35% of cases the first attempt already renamed. The transformation is
#      identical to theirs: strip everything up to and including the last '-'.
# ============================================================================

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate segmamba

SEGV2_SRC=/path/to/home/AGILEMamba/SegMamba-V2/brats23
SRC_RAW="/path/to/home/AGILEMamba/BraTS 2023 offi/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
WORK=/path/to/scratch/segv2_repro
RAWDST="$WORK/data/raw_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

mkdir -p "$RAWDST"
echo "=== scratch free space before ==="
df -h /mnt/scratch | tail -1

# --- bring their pipeline code alongside the data --------------------------
# Their scripts use relative paths (./data/raw_data, ./data/fullres), so the
# code has to sit next to the data and we have to cd into $WORK.
for d in light_training models_segmamba; do
    if [ ! -e "$WORK/$d" ]; then
        cp -r "$SEGV2_SRC/$d" "$WORK/"
    fi
done
cp "$SEGV2_SRC/2_preprocessing_BraTS2023.py" "$WORK/"
mkdir -p "$WORK/data"
cp "$SEGV2_SRC/data/test_list.py" "$WORK/data/" 2>/dev/null || true

# --- copy raw data (once only) ----------------------------------------------
# GATED BY A MARKER, deliberately. --ignore-existing is NOT safe to re-run here:
# once the rename has turned BraTS-GLI-XXXXX-000-t2w.nii.gz into t2w.nii.gz,
# rsync sees the long name as missing and copies it back, doubling the files in
# every already-renamed case. (Observed in job 1881451: 10595 files instead of
# 6255.) The rename then collapses them again, so it is harmless but pure waste.
COPY_MARKER="$WORK/.raw_copy_complete"
echo "=== 0) copying raw cases to scratch ==="
if [ -f "$COPY_MARKER" ]; then
    echo "raw copy already complete (marker present), skipping rsync"
else
    echo "source entries: $(ls "$SRC_RAW" | wc -l)"
    echo "source dirs:    $(find "$SRC_RAW" -maxdepth 1 -mindepth 1 -type d | wc -l)  (expect 1251)"
    rsync -a --ignore-existing "$SRC_RAW/" "$RAWDST/"
    echo "--- removing loose non-directory entries (stray .pth checkpoints) ---"
    find "$RAWDST" -maxdepth 1 -mindepth 1 -not -type d -print -delete || true
    touch "$COPY_MARKER"
fi
echo "copied case dirs: $(find "$RAWDST" -maxdepth 1 -mindepth 1 -type d | wc -l)"
du -sh "$RAWDST"

# --- rename step (idempotent; same transformation as their 1_rename) --------
echo "=== 1) rename modalities in place (scratch copy only) ==="
cd "$WORK"
python - <<'PY'
import os

root = ("./data/raw_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
KEEP = {"t1c.nii.gz", "t1n.nii.gz", "t2f.nii.gz", "t2w.nii.gz", "seg.nii.gz"}

renamed = skipped = 0
for case in sorted(os.listdir(root)):
    case_dir = os.path.join(root, case)
    if not os.path.isdir(case_dir):
        continue
    for fn in os.listdir(case_dir):
        if fn in KEEP:            # already renamed by the failed first attempt
            skipped += 1
            continue
        # Their transformation: BraTS-GLI-00045-000-seg.nii.gz -> seg.nii.gz
        new = fn.split("-")[-1]
        if new == fn:
            continue
        os.rename(os.path.join(case_dir, fn), os.path.join(case_dir, new))
        renamed += 1
print(f"renamed {renamed}, already-correct {skipped}")
PY

echo "=== sanity: contents of one case after rename ==="
# NB: NOT `ls | head -1`. head exits after one line, ls dies of SIGPIPE, and
# `set -o pipefail` turns that into a job failure -- exit code 13 (= SIGPIPE)
# is what killed jobs 1881451 and 1881458, both AFTER the rename had already
# succeeded. find -quit avoids the pipe entirely.
first_case=$(find "$RAWDST" -maxdepth 1 -mindepth 1 -type d -printf '%p\n' -quit)
echo "$first_case"
ls "$first_case"
# The count below is done in python, not a bash for-loop. A loop ending in a
# false [ ] test
# returns non-zero, which `set -e` treats as a job failure -- that is what
# killed job 1881451 *after* the rename had already succeeded.
python - <<'PY'
import os
root = "./data/raw_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
NEED = {"t1c.nii.gz", "t1n.nii.gz", "t2f.nii.gz", "t2w.nii.gz", "seg.nii.gz"}
ok, bad = 0, []
for case in sorted(os.listdir(root)):
    d = os.path.join(root, case)
    if not os.path.isdir(d):
        continue
    files = set(os.listdir(d))
    if files == NEED:
        ok += 1
    else:
        bad.append((case, sorted(files - NEED), sorted(NEED - files)))
print(f"cases with exactly the 5 expected files: {ok}  (expect 1251)")
for case, extra, missing in bad[:10]:
    print(f"  BAD {case}  extra={extra}  missing={missing}")
if bad:
    raise SystemExit(f"{len(bad)} malformed case dirs -- aborting before preprocessing")
PY

# --- their preprocessing ----------------------------------------------------
echo "=== 2) preprocessing (plan + process_train) ==="
cd "$WORK"
python "$WORK/2_preprocessing_BraTS2023.py"

echo "=== preprocessed output ==="
ls "$WORK/data/fullres/train" | head || true   # same SIGPIPE hazard as above
echo "npz count: $(ls "$WORK/data/fullres/train"/*.npz 2>/dev/null | wc -l)  (expect 1251)"
du -sh "$WORK/data/fullres"
echo "=== scratch free space after ==="
df -h /mnt/scratch | tail -1
echo "=== home quota (should be UNCHANGED) ==="
quota -s || true
