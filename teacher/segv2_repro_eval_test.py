# -*- coding: utf-8 -*-
"""
Evaluate the REPRODUCED SegMamba-V2 (trained on our 876, checkpoint selected
on our 125 val) on our 250-case test set -- the first genuinely like-for-like
comparison against AGILE-Mamba.

WHY THIS SCRIPT EXISTS: their trainer's own validation metric scores raw
classes (labels 1/2/3 = NCR/ED/ET), which is why its checkpoints are named
~0.83-0.89. Their PAPER reports regions (WT/TC/ET), which are unions of those
classes and therefore score several points higher on the same prediction.
Nothing in their pipeline computes the region numbers during training, so we
compute them here, once, on the test set.

Scoring is deliberately identical to what AGILE-Mamba was scored with:
    dice = (2*inter + eps) / (denom + eps),  eps = 1e-5
which also matches their cal_metric convention (both-empty -> 1.0,
empty-GT-with-prediction -> ~0.0).

Inference is deliberately identical to THEIRS: sliding window at 128^3,
sw_batch_size 2, overlap 0.5, gaussian weighting, mirror TTA on all 3 axes.

REGION ORDER -- the trap in this codebase:
    theirs (convert_labels): [TC, WT, ET]
    ours   (dice_score)    : [WT, TC, ET]
Getting this wrong silently swaps WT and TC, which is hard to spot because
both are high. We reorder theirs to ours before scoring.

Run from /path/to/scratch/segv2_repro (their code uses relative
imports of light_training / models_segmamba).
"""
import os
import sys
import glob

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monai.inferers import SlidingWindowInferer
from light_training.dataloading.dataset import MedicalDataset
from light_training.prediction import Predictor
from models_segmamba.segmambav2 import SegMamba

WORK = "/path/to/scratch/segv2_repro"
DATA_DIR = f"{WORK}/data/fullres/train"
SPLIT_DIR = f"{WORK}/splits"
MODEL_DIR = f"{WORK}/logs/segmambav2_ourssplit/model"

PATCH = [128, 128, 128]
DEVICE = "cuda:0"
EPS = 1e-5


def dice_ours(pred, target):
    """Our formula, per region. pred/target: (K, D, H, W) float tensors."""
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


def pick_checkpoint():
    """Prefer best_model_*.pt (selected on our 125-case val set).

    NOT final_model_*.pt: that is simply the last epoch. Their released
    checkpoint happens to be a final_model, but 'best on validation' is the
    convention this project uses everywhere else, and is what the comparison
    against AGILE-Mamba's best_model.pt requires.
    """
    cands = sorted(glob.glob(os.path.join(MODEL_DIR, "best_model_*.pt")))
    if not cands:
        raise FileNotFoundError(f"no best_model_*.pt in {MODEL_DIR}")
    if len(cands) > 1:
        print(f"WARNING: {len(cands)} best_model files, using the last: {cands}")
    return cands[-1]


def main():
    with open(os.path.join(SPLIT_DIR, "test_ids.txt")) as f:
        test_ids = [l.strip() for l in f if l.strip()]
    print(f"test cases: {len(test_ids)}  (expect 250)")
    assert len(test_ids) == 250

    all_paths = sorted(glob.glob(f"{DATA_DIR}/*.npz"))
    by_id = {p.split("/")[-1].split(".")[0]: p for p in all_paths}
    missing = [i for i in test_ids if i not in by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} test cases not preprocessed: {missing[:5]}")
    test_ds = MedicalDataset([by_id[i] for i in test_ids])

    ckpt = pick_checkpoint()
    print(f"checkpoint: {ckpt}")
    model = SegMamba(4, 4)
    model.load_state_dict(filter_state_dict(torch.load(ckpt, map_location="cpu")))
    model.eval().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SegMamba-V2 (reproduced on our split): {n_params/1e6:.2f}M parameters")

    window_infer = SlidingWindowInferer(roi_size=PATCH, sw_batch_size=2,
                                        overlap=0.5, progress=False, mode="gaussian")
    predictor = Predictor(window_infer=window_infer, mirror_axes=[0, 1, 2])

    all_d, names = [], []
    for i in range(len(test_ds)):
        batch = test_ds[i]
        # Their MedicalDataset returns numpy memmaps, not tensors.
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
    print("SegMamba-V2 REPRODUCED on our split -- 250-case test set")
    print("  trained on our 876, checkpoint selected on our 125 val")
    print("  region order [WT, TC, ET], our dice formula, their inference + mirror TTA")
    print("=" * 70)
    print(f"  WT {100*m[0]:.2f}   TC {100*m[1]:.2f}   ET {100*m[2]:.2f}")
    print(f"  MEAN DICE: {100*m.mean():.2f}%")
    print()
    print("  AGILE-Mamba on the same 250 cases: 90.10%  (17.70M params)")
    print(f"  SegMamba-V2 reproduced:           {100*m.mean():.2f}%  ({n_params/1e6:.2f}M params)")
    print("  SegMamba-V2 as published (their split, their training data): 91.60%")

    np.save(os.path.join(WORK, "segv2_repro_test250_percase.npy"), D.numpy())
    with open(os.path.join(WORK, "segv2_repro_test250_cases.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    print("\n  per-case Dice saved -- enables a paired test against AGILE-Mamba's per-case numbers")


if __name__ == "__main__":
    main()
