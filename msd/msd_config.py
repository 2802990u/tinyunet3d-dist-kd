# msd_config.py
# Shared per-task config for the MSD small-benchmark pipeline (single-teacher paper).
# Every MSD script reads TASK from the environment (default "spleen") and pulls its
# settings from here, so the stager / preprocessor / teacher / student / eval all
# agree by construction.
#
# Tasks (Medical Segmentation Decathlon, http://medicaldecathlon.com):
#   spleen      — Task09_Spleen,      41 labelled CT,  1 fg class  (small/label-scarce point)
#   hippocampus — Task04_Hippocampus, 260 labelled MRI, 2 fg classes (large-N, tiny structures)
#   heart       — Task02_Heart,       20 labelled MRI, 1 fg class  (left atrium)
#   prostate    — Task05_Prostate,    32 labelled MRI (2-channel), 2 fg classes (OPTIONAL — see caveat)
#
# AWS mirrors (no Google Drive login needed), download into ./raw/ and untar:
#   https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar
#   https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar
#   https://msd-for-monai.s3-us-west-2.amazonaws.com/Task02_Heart.tar
#   https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar
#
# PROSTATE CAVEAT: MSD prostate images are 4-D NIfTI (2 modalities in one file).
# The base light_training DefaultPreprocessor assumes a single 3-D image per case —
# verify it handles the 4-D volume (or split channels) before trusting prostate runs.
# Spleen + Heart are plain 3-D and safe.
import os

# Split convention for the small MSD tasks: fixed seeded 70/10/20 via
# get_train_val_test_loader_from_train(seed=42) — select on val, report on test.
# (BraTS-full used 80/10/10, but with only 20-41 cases a 10% test set is 2-4 cases;
# 20% keeps the held-out report meaningful. Same rates for every task and method.)
SPLIT_SEED, TRAIN_RATE, VAL_RATE, TEST_RATE = 42, 0.7, 0.1, 0.2

TASKS = {
    "spleen": dict(
        task_dir="Task09_Spleen",
        data_type="CT",              # CTNormalization (foreground stats), like AMOS
        in_ch=1, out_ch=2,           # bg + spleen
        class_names=["spleen"],
        # nnU-Net 3d_fullres spleen patch; anisotropic abdominal CT.
        patch_size=[64, 192, 160],
        file_prefix="spleen_",
    ),
    "heart": dict(
        task_dir="Task02_Heart",
        data_type="MRI",             # per-image Z-score (verify DefaultPreprocessor's MRI branch)
        in_ch=1, out_ch=2,           # bg + left atrium
        class_names=["left_atrium"],
        patch_size=[80, 192, 160],   # nnU-Net 3d_fullres heart patch
        file_prefix="la_",
    ),
    "hippocampus": dict(
        task_dir="Task04_Hippocampus",
        data_type="MRI",             # mono-modal T1; per-image Z-score
        in_ch=1, out_ch=3,           # bg + anterior + posterior hippocampus
        class_names=["anterior", "posterior"],
        # Volumes are TINY (~36x50x35 at 1mm iso). nnU-Net uses [40,56,40], but
        # SegMamba downsamples /16 so every dim must be divisible by 16.
        # [32,48,32] is the largest such patch that still fits INSIDE most
        # volumes -> minimal padding (see the spleen padding lesson).
        patch_size=[32, 48, 32],
        file_prefix="hippocampus_",
    ),
    "acdc": dict(
        task_dir="ACDC",
        data_type="MRI",             # cine short-axis MRI; per-image Z-score
        in_ch=1, out_ch=4,           # bg + RV cavity(1) + myocardium(2) + LV cavity(3)
        class_names=["RV", "myo", "LV"],   # ACDC label convention: 1=RV, 2=Myo, 3=LV
        # ACDC short-axis stacks are SHALLOW (~6-18 slices) and wide in-plane.
        # SegMamba downsamples /16 so every dim must be divisible by 16.
        # Same lesson as hippocampus: pick the largest /16 patch that still fits
        # INSIDE most volumes so the loader pads as little as possible.
        # [16,128,128] mirrors the (10,128,128) patch that worked for ACDC in the
        # LWSegformer3D project, rounded up to the nearest /16 on the z axis.
        # Check "median_shape" printed by run_plan() and adjust if it disagrees.
        patch_size=[16, 128, 128],
        file_prefix="acdc_",
    ),
    "prostate": dict(
        task_dir="Task05_Prostate",
        data_type="MRI",
        in_ch=2, out_ch=3,           # bg + peripheral zone + transition zone
        class_names=["PZ", "TZ"],
        # nnU-Net uses [20,320,256]; SegMamba's encoder downsamples /16, so depth
        # must be divisible by 16 -> use 32 (loader pads shallow volumes).
        patch_size=[32, 320, 256],
        file_prefix="prostate_",
    ),
}


def get_task():
    name = os.environ.get("TASK", "spleen").lower()
    if name not in TASKS:
        raise KeyError(f"TASK={name!r} — must be one of {sorted(TASKS)}")
    cfg = dict(TASKS[name])
    cfg["name"] = name
    cfg["raw_dir"] = os.path.abspath(os.path.join("./raw", cfg["task_dir"]))
    return cfg
