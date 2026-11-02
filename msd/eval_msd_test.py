# eval_msd_test.py
# Report FINAL numbers on the held-out MSD test split (fixed 70/10/20, seed=42).
#
#   TASK       = spleen | heart | prostate
#   MODEL_PATH = path/glob to a trained best_model_*.pt
#   ARCH       = student | segmamba   (default student)
#   STUDENT_BASE = 32 | 48 | 64       (must match the trained run)
#   SPLIT      = test | val           (default test; use val to CHOOSE settings
#                                      without touching the test set)
#
# --- SLIDING-WINDOW PADDING (important) -------------------------------------
# MONAI pads volumes smaller than the patch. Its default is constant 0, but after
# CT normalization 0 is ~mean FOREGROUND intensity (air is ~-3.3), so the padded
# region looks like solid tissue. On MSD Spleen this made the SegMamba teacher
# predict pure background on a 48-slice volume (Dice 0.0000) while scoring
# 0.92-0.97 everywhere else. Half the spleen test cases are shallower than the
# 64-slice patch, so this is not an edge case.
#
# Default here is the physically correct choice: constant padding with the
# volume's own air value (per-case min of the normalized image), i.e. "outside
# the field of view is air". Overridable:
#   SW_PAD_MODE = constant | replicate     (default constant)
#   SW_CVAL     = auto | <float>           (default auto -> per-case image min)
#   SW_BLEND    = constant | gaussian      (default constant)
#   SW_OVERLAP  = float                    (default 0.5)
# Use the SAME settings for every model being compared.
#
# Run from the task working dir:
#   TASK=spleen ARCH=student STUDENT_BASE=64 MODEL_PATH='./logs/student_spleen_kl1_b64_s1/model/best_model_*.pt' python eval_msd_test.py
#   TASK=spleen ARCH=segmamba MODEL_PATH='./logs/ssm_teacher_spleen/model/best_model_*.pt' python eval_msd_test.py
import os, glob
import numpy as np
import torch
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.evaluation.metric import dice
from monai.inferers import SlidingWindowInferer
from msd_config import get_task, SPLIT_SEED, TRAIN_RATE, VAL_RATE, TEST_RATE

CFG        = get_task()
patch_size = CFG["patch_size"]
IN_CH, OUT_CH = CFG["in_ch"], CFG["out_ch"]
N_FG       = OUT_CH - 1
STUDENT_BASE = int(os.environ.get("STUDENT_BASE", "32"))
ARCH       = os.environ.get("ARCH", "student").lower()
SPLIT      = os.environ.get("SPLIT", "test").lower()
MODEL_PATH = os.environ["MODEL_PATH"]      # required (glob patterns allowed)

SW_PAD_MODE = os.environ.get("SW_PAD_MODE", "constant").lower()
SW_CVAL     = os.environ.get("SW_CVAL", "auto").lower()
SW_BLEND    = os.environ.get("SW_BLEND", "constant").lower()
SW_OVERLAP  = float(os.environ.get("SW_OVERLAP", "0.5"))

data_dir   = "./data/fullres/train"

_matches = sorted(glob.glob(MODEL_PATH))
if _matches:
    MODEL_PATH = _matches[-1]


def filter_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    return {(k[7:] if str(k).startswith("module") else str(k)): v for k, v in sd.items()}


def build_model():
    if ARCH == "student":
        from student_unet3d import TinyUNet3D
        return TinyUNet3D(in_chans=IN_CH, out_chans=OUT_CH, base=STUDENT_BASE)
    elif ARCH == "segmamba":
        from models_segmamba.segmambav2 import SegMamba
        return SegMamba(IN_CH, OUT_CH)
    raise ValueError(f"ARCH={ARCH!r} — must be student | segmamba")


def make_inferer(image):
    """Per-case inferer: 'auto' cval uses this volume's air value (its min)."""
    cval = float(image.min()) if SW_CVAL == "auto" else float(SW_CVAL)
    return SlidingWindowInferer(roi_size=patch_size,
                                sw_batch_size=(4 if ARCH == "student" else 2),
                                overlap=SW_OVERLAP,
                                mode=SW_BLEND,
                                padding_mode=SW_PAD_MODE,
                                cval=cval)


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        return float(dice(pred, gt))
    elif gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    return 0.0


if __name__ == "__main__":
    tr_ds, val_ds, test_ds = get_train_val_test_loader_from_train(
        data_dir=data_dir, train_rate=TRAIN_RATE, val_rate=VAL_RATE, test_rate=TEST_RATE,
        seed=SPLIT_SEED)
    ds = {"test": test_ds, "val": val_ds}[SPLIT]

    model = build_model()
    model.load_state_dict(filter_state_dict(torch.load(MODEL_PATH, map_location="cpu")))
    model = model.to("cuda:0").eval()

    print(f"Evaluating [{ARCH}] {MODEL_PATH}")
    print(f"  split={SPLIT} ({len(ds)} cases)  patch={patch_size}")
    print(f"  sliding-window: pad={SW_PAD_MODE} cval={SW_CVAL} blend={SW_BLEND} overlap={SW_OVERLAP}")

    records = []
    for i in range(len(ds)):
        it = ds[i]
        image = torch.as_tensor(np.asarray(it["data"])).unsqueeze(0).to("cuda:0").float()
        target = np.asarray(it["seg"])[0]
        with torch.no_grad():
            out = make_inferer(image)(image, model).argmax(dim=1)[0].cpu().numpy()
        d = [cal_metric(target == c, out == c) for c in range(1, OUT_CH)]
        records.append(d)
        print(f"  case {i+1:03d}: mean={np.mean(d):.4f}"
              + (f"   pred_fg={int((out>0).sum()):,}" if N_FG == 1 else ""))
        del image
        torch.cuda.empty_cache()

    arr = np.array(records)                 # (N, N_FG)
    per_class = arr.mean(axis=0)
    per_case_mean = arr.mean(axis=1)
    print(f"\n================ {CFG['name'].upper()} {SPLIT.upper()} RESULT ================")
    print(f"arch: {ARCH}   base: {STUDENT_BASE if ARCH=='student' else '-'}   model: {MODEL_PATH}")
    print(f"padding: {SW_PAD_MODE}/{SW_CVAL}  blend: {SW_BLEND}  overlap: {SW_OVERLAP}")
    print("per-class mean Dice:  " + "  ".join(
        f"{CFG['class_names'][i]}={per_class[i]:.4f}" for i in range(N_FG)))
    print(f"overall mean Dice: {per_case_mean.mean():.4f}  "
          f"(per-case std {per_case_mean.std():.4f}, N={len(arr)})")
    print("=====================================================")
