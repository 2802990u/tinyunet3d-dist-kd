# fixed_split_loader.py
# Fixed-instance BraTS 2023 split shared by teacher and student: 876 train /
# 125 val / 250 test, read from the same id files the teacher's
# segv2_repro_train.py / segv2_repro_eval_test.py already use. Drop-in
# replacement for get_train_val_test_loader_from_train, minus the rate/seed
# arguments -- there is now exactly one source of truth for split membership.
#
# Place this file in SegMamba-V2/brats23/ and symlink it into
# SegMamba-V2/analysis/ (the eval scripts run from there):
#   ln -s ../brats23/fixed_split_loader.py ../analysis/fixed_split_loader.py
import os
import glob

from light_training.dataloading.dataset import MedicalDataset

SPLIT_DIR = os.environ.get("SPLIT_DIR", "./splits")


def _read_ids(split_dir, name):
    with open(os.path.join(split_dir, f"{name}.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def get_fixed_split_loader(data_dir, split_dir=SPLIT_DIR):
    """Returns (train_ds, val_ds, test_ds) as MedicalDataset objects, built
    from the fixed 876/125/250 id lists rather than a fresh random shuffle.

    Raises if any id in the split files has no matching preprocessed .npz in
    data_dir, and asserts the split sizes and non-overlap, so a silent
    mismatch (wrong data_dir, stale split files, etc.) fails loudly instead
    of quietly training on the wrong cases.
    """
    all_paths = sorted(glob.glob(f"{data_dir}/*.npz"))
    by_id = {p.split("/")[-1].split(".")[0]: p for p in all_paths}

    out = {}
    for name in ("train_ids", "val_ids", "test_ids"):
        ids = _read_ids(split_dir, name)
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise RuntimeError(
                f"{len(missing)} cases in {name}.txt have no preprocessed .npz "
                f"in {data_dir}, e.g. {missing[:5]}")
        out[name] = [by_id[i] for i in ids]

    n_tr, n_va, n_te = len(out["train_ids"]), len(out["val_ids"]), len(out["test_ids"])
    assert (n_tr, n_va, n_te) == (876, 125, 250), \
        f"split sizes are wrong: got {(n_tr, n_va, n_te)}, expected (876, 125, 250)"
    s_tr, s_va, s_te = set(out["train_ids"]), set(out["val_ids"]), set(out["test_ids"])
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), \
        "split overlap detected -- this must never happen"

    train_ds = MedicalDataset(out["train_ids"])
    val_ds   = MedicalDataset(out["val_ids"])
    test_ds  = MedicalDataset(out["test_ids"])
    return train_ds, val_ds, test_ds
