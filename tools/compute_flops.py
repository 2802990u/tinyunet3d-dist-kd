# compute_flops.py
# Computes real FLOPs/params for TinyUNet3D (the student) at each dataset's
# actual input patch size, using thop. Run this on the HPC, not locally --
# it needs your real student_unet3d.py / msd_config.py etc. to build the
# exact model you trained.
#
# Usage:
#   pip install thop --break-system-packages   # one-time
#   python compute_flops.py                    # runs all 4 datasets
#   DATASETS=spleen,brats python compute_flops.py   # just a subset
#
# IMPORTANT CAVEAT: thop reports MACs (multiply-accumulate ops), not FLOPs.
# The convention in most papers is FLOPs = 2 x MACs (one multiply + one add
# per MAC) -- this script prints both so you can pick whichever convention
# your paper/related-work table uses, but make sure it's consistent with
# whatever the SOTA papers in your comparison table report, or the numbers
# aren't apples-to-apples.
#
# TEACHER (SegMamba-V2) IS NOT COVERED HERE. thop/fvcore do not reliably
# count FLOPs through custom CUDA selective-scan (Mamba) kernels -- they'll
# either error or silently undercount those layers. If you need a teacher
# FLOPs number, check whether the SegMamba-V2 paper itself reports one
# (cite theirs) rather than trusting an automatic profiler on a Mamba model.

import os
import sys
import importlib

try:
    from thop import profile
except ImportError:
    raise SystemExit("Run: pip install thop --break-system-packages")

import torch

STARTING_CWD = os.getcwd()
MSD_ROOT = os.path.expanduser(os.environ.get("MSD_ROOT", "~/SingleTeacherKD"))
SEGMAMBA_ROOT = os.path.expanduser(os.environ.get("SEGMAMBA_ROOT", "~/CoPa-Distill/SegMamba-V2"))

DATASET_DIRS = {
    "spleen": os.path.join(MSD_ROOT, "msd_spleen"),
    "hippocampus": os.path.join(MSD_ROOT, "msd_hippocampus"),
    "amos": os.path.join(SEGMAMBA_ROOT, "amos_ct"),
    "brats": os.path.join(SEGMAMBA_ROOT, "brats23"),
}

ORDER_ALL = ["spleen", "hippocampus", "amos", "brats"]
ORDER = os.environ.get("DATASETS", ",".join(ORDER_ALL)).split(",")


def clear_module_cache():
    for name in list(sys.modules):
        if any(name.startswith(p) for p in
               ("msd_config", "student_unet3d", "models_segmamba", "light_training", "data", "acdc_config")):
            del sys.modules[name]


def enter_dir(path):
    os.chdir(path)
    sys.path.insert(0, path)


def leave_dir(path):
    if path in sys.path:
        sys.path.remove(path)
    os.chdir(STARTING_CWD)


# --- FILL THIS IN per dataset, copying IN_CH/OUT_CH/patch_size exactly
# from whatever each dataset's own train_student_*.py or msd_config.py
# actually passes to TinyUNet3D(...) -- do not guess these, they must
# match training exactly or the FLOPs number is for the wrong model.
DATASET_MODEL_ARGS = {
    "spleen":      dict(IN_CH=1, OUT_CH=2,  patch_size=(64, 192, 160)),
    "hippocampus": dict(IN_CH=1, OUT_CH=3,  patch_size=(32, 48, 32)),
    "amos":        dict(IN_CH=1, OUT_CH=16, patch_size=(80, 160, 160)),
    "brats":       dict(IN_CH=4, OUT_CH=3,  patch_size=(128, 128, 128)),
}


def compute_for(key):
    path = DATASET_DIRS[key]
    clear_module_cache()
    enter_dir(path)
    try:
        from student_unet3d import TinyUNet3D  # adjust class/module name if different

        args = DATASET_MODEL_ARGS[key]
        patch_size = args["patch_size"]
        model = TinyUNet3D(in_channels=args["IN_CH"], out_channels=args["OUT_CH"]).eval()
        dummy = torch.randn(1, args["IN_CH"], *patch_size)

        macs, params = profile(model, inputs=(dummy,), verbose=False)
        flops = 2 * macs
        return {
            "patch_size": patch_size,
            "params_M": params / 1e6,
            "MACs_G": macs / 1e9,
            "FLOPs_G": flops / 1e9,
        }
    finally:
        leave_dir(path)


if __name__ == "__main__":
    print(f"{'Dataset':<14}{'Patch size':<18}{'Params (M)':<12}{'MACs (G)':<12}{'FLOPs (G)':<12}")
    for key in ORDER:
        key = key.strip()
        try:
            r = compute_for(key)
            print(f"{key:<14}{str(r['patch_size']):<18}{r['params_M']:<12.3f}{r['MACs_G']:<12.3f}{r['FLOPs_G']:<12.3f}")
        except Exception as e:
            print(f"{key:<14}FAILED: {e}")
