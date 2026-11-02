# -*- coding: utf-8 -*-
"""
Export OUR fixed 70/10/20 split (split_seed=42, 876/125/250) into plain text
lists that SegMamba-V2's dataloader can consume, so their model is trained,
validated and tested on exactly the same partition as AGILE-Mamba.

Writes into the scratch working dir:
    /path/to/scratch/segv2_repro/splits/train_ids.txt   (876)
    /path/to/scratch/segv2_repro/splits/val_ids.txt     (125)
    /path/to/scratch/segv2_repro/splits/test_ids.txt    (250)

WHY THIS MATTERS -- the protocol deviation we are deliberately making:
their 3_train.py sets `val_dataset = test_ds`, i.e. it selects the best
checkpoint on the test set. That is fine for reproducing their headline
number in their own paper's terms, but if we did it here their model would
get to peek at OUR 250 test cases while AGILE-Mamba never did, and the
comparison would be worthless in our favour's opposite direction. So:

    train on our 876   |   select checkpoint on our 125 val   |   test once on our 250

Everything else stays theirs exactly: SGD 1e-2 / momentum 0.99 nesterov /
wd 3e-5, poly schedule, CE-only loss, 1000 epochs, 128^3 patches, their
augmentation, sliding-window inference with mirror TTA, SegMamba(4, 4).

We also emit a `best_of_val` note: their paper reports the best validation
epoch rather than a held-out test number. We record BOTH for our model and
theirs so the paper can report like-for-like whichever convention we adopt.
"""
import os
import sys

sys.path.insert(0, "/path/to/home/AGILEMamba")

from brats_dataset import make_single_split  # our canonical 70/10/20 splitter

DATA_ROOT = ("/path/to/home/AGILEMamba/BraTS 2023 offi/"
             "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
OUT = "/path/to/scratch/segv2_repro/splits"

# Cross-check against the split actually used by our best model, so a silent
# change in make_split() can never go unnoticed.
REFERENCE_SPLIT_DIR = ("/path/to/home/AGILEMamba/"
                       "checkpoints_singlesplit_base17m_bidir_c2c3")


def read_ids(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def main():
    os.makedirs(OUT, exist_ok=True)

    train_ids, val_ids, test_ids = make_single_split(DATA_ROOT, seed=42)
    print(f"train {len(train_ids)}  val {len(val_ids)}  test {len(test_ids)}")
    assert (len(train_ids), len(val_ids), len(test_ids)) == (876, 125, 250), \
        "split sizes do not match the 876/125/250 protocol"
    assert not (set(train_ids) & set(val_ids)), "train/val overlap"
    assert not (set(train_ids) & set(test_ids)), "train/test overlap"
    assert not (set(val_ids) & set(test_ids)), "val/test overlap"

    # Hard verification against the split our reported model actually used.
    for name, ids in (("val_ids", val_ids), ("test_ids", test_ids)):
        ref = os.path.join(REFERENCE_SPLIT_DIR, f"{name}.txt")
        if os.path.exists(ref):
            got = set(read_ids(ref))
            assert got == set(ids), (
                f"{name} differs from {ref} -- the reproduction would not be "
                f"on the same split as our reported result")
            print(f"  verified {name} matches {ref}")
        else:
            print(f"  WARNING: {ref} not found, could not cross-check {name}")

    for name, ids in (("train_ids", train_ids),
                      ("val_ids", val_ids),
                      ("test_ids", test_ids)):
        p = os.path.join(OUT, f"{name}.txt")
        with open(p, "w") as f:
            f.write("\n".join(sorted(ids)) + "\n")
        print(f"  wrote {p}  ({len(ids)})")

    # Their preprocessed files are named by case id; confirm every id we are
    # about to ask for actually exists in the fullres output.
    fullres = "/path/to/scratch/segv2_repro/data/fullres/train"
    if os.path.isdir(fullres):
        have = {f[:-4] for f in os.listdir(fullres) if f.endswith(".npz")}
        missing = [i for i in (train_ids + val_ids + test_ids) if i not in have]
        print(f"  fullres cases present: {len(have)}")
        if missing:
            print(f"  MISSING {len(missing)} preprocessed cases, e.g. {missing[:5]}")
        else:
            print("  all 1251 split cases present in fullres")
    else:
        print(f"  (fullres dir not created yet: {fullres})")


if __name__ == "__main__":
    main()
