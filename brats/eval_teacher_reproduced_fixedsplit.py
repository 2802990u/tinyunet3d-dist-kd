# eval_teacher_reproduced_fixedsplit.py
# Evaluate the REPRODUCED SegMamba-V2 checkpoint (best_model_0.8921.pt --
# trained from scratch on our own 876-case train split, selected on our own
# 125-case val, never touching our 250-case test set) on that fixed 250-case
# test split. This -- not the officially released checkpoint -- is the
# leak-free "Teacher (ours)" number for the paper: the released checkpoint
# scored 93.29% on this same test set, higher than its own paper's published
# 91.60%, which means it had already seen many of these cases during its
# original training.
#
# Scoring/inference deliberately mirror segv2_repro_eval_test.py: sliding
# window at 128^3, overlap 0.5, gaussian weighting, mirror TTA on all 3 axes,
# region order [WT, TC, ET] (their convert_labels gives [TC, WT, ET] -- this
# script reorders explicitly, same trap flagged in that script).
#
# Run from SegMamba-V2/brats23/:
#   python eval_teacher_reproduced_fixedsplit.py
#
# NOTE: this assumes light_training.prediction.Predictor exists in this
# brats23 copy -- confirmed working already, since
# eval_teacher_released_fixedsplit.py (same import) ran successfully.
import os
import numpy as np
import torch

from fixed_split_loader import get_fixed_split_loader
from monai.inferers import SlidingWindowInferer
from light_training.prediction import Predictor
from models_segmamba.segmambav2 import SegMamba

data_dir = os.environ.get("BRATS_DATA", "./data/brats_fullres/train")
CKPT     = os.environ.get("TEACHER_CKPT", "./segmambav2_teacher_reproduced.pt")
IN_CH, OUT_CH = 4, 4
PATCH = [128, 128, 128]
DEVICE = "cuda:0"
EPS = 1e-5


def dice_ours(pred, target):
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    return (2 * inter + EPS) / (denom + EPS)


def convert_labels_theirs(labels):
    """Their mapping, on dim 0 -> [TC, WT, ET]."""
    return torch.cat([
        (labels == 1) | (labels == 3),                    # TC
        (labels == 1) | (labels == 3) | (labels == 2),    # WT
        (labels == 3),                                    # ET
    ], dim=0).float()


def filter_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    out = {}
    for k, v in sd.items():
        k = str(k)
        out[k[7:] if k.startswith("module") else k] = v
    return out


def main():
    _tr, _val, test_ds = get_fixed_split_loader(data_dir)
    print(f"test cases: {len(test_ds)}  (expect 250)")
    assert len(test_ds) == 250

    model = SegMamba(IN_CH, OUT_CH)
    model.load_state_dict(filter_state_dict(torch.load(CKPT, map_location="cpu")))
    model.eval().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SegMamba-V2 (reproduced checkpoint, {CKPT}): {n_params/1e6:.2f}M parameters")

    window_infer = SlidingWindowInferer(roi_size=PATCH, sw_batch_size=2,
                                        overlap=0.5, progress=False, mode="gaussian")
    predictor = Predictor(window_infer=window_infer, mirror_axes=[0, 1, 2])

    all_d, names = [], []
    for i in range(len(test_ds)):
        batch = test_ds[i]
        image = torch.from_numpy(np.array(batch["data"])).float()
        label = torch.from_numpy(np.array(batch["seg"])).float()
        props = batch["properties"]
        name = props["name"] if isinstance(props["name"], str) else props["name"][0]

        if image.ndim == 4:
            image = image[None]

        with torch.no_grad():
            out = predictor.maybe_mirror_and_predict(image, model, device=DEVICE)
            out = predictor.predict_raw_probability(out, properties=props)

        pred = out.argmax(dim=0)[None]
        pred_regions = convert_labels_theirs(pred)     # [TC, WT, ET]
        lab = label if label.ndim == 4 else label[None]
        lab_regions = convert_labels_theirs(lab)       # [TC, WT, ET]

        # reorder both to OUR convention [WT, TC, ET]
        pred_regions = pred_regions[[1, 0, 2]]
        lab_regions = lab_regions[[1, 0, 2]]

        d = dice_ours(pred_regions.cpu().float(), lab_regions.cpu().float())
        all_d.append(d)
        names.append(name)
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  {i+1:3d}/{len(test_ds)}  ({name})  "
                  f"WT {d[0]:.4f}  TC {d[1]:.4f}  ET {d[2]:.4f}", flush=True)

    D = torch.stack(all_d)
    m = D.mean(dim=0)
    print("=" * 70)
    print("SegMamba-V2 REPRODUCED CHECKPOINT on the fixed 250-case test set")
    print("  (trained on our 876, selected on our 125 val -- NEVER touched test)")
    print("  region order [WT, TC, ET], our dice formula, their inference + mirror TTA")
    print("=" * 70)
    print(f"  WT {100*m[0]:.2f}   TC {100*m[1]:.2f}   ET {100*m[2]:.2f}")
    print(f"  MEAN DICE: {100*m.mean():.2f}%")

    np.save("./teacher_reproduced_fixedsplit_test250_percase.npy", D.numpy())
    with open("./teacher_reproduced_fixedsplit_test250_cases.txt", "w") as f:
        f.write("\n".join(names) + "\n")
    print("\n  per-case Dice saved -- this is your leak-free Teacher (ours) row for the paper.")


if __name__ == "__main__":
    main()
