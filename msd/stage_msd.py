# stage_msd.py
# Stage 1a for an MSD task: expose the labelled training set in the layout the
# light_training DefaultPreprocessor expects (base_dir/imagesTr_clean +
# base_dir/labelsTr_clean, same filenames), filtering the macOS "._*" junk files
# that ship inside the MSD tarballs. Non-destructive: symlinks only.
#
# No test_list is written — like brats_full, the split is the fixed seeded
# 70/10/20 from get_train_val_test_loader_from_train(seed=42) at train time.
#
# Run from the task working dir (e.g. ~/CoPa-Distill/SegMamba-V2/msd_spleen):
#   TASK=spleen python stage_msd.py
# expects ./raw/Task09_Spleen/{imagesTr,labelsTr} (download + untar first — see msd_config.py).
import os
from msd_config import get_task

CFG      = get_task()
RAW      = CFG["raw_dir"]
IMG_SRC  = os.path.join(RAW, "imagesTr")
LBL_SRC  = os.path.join(RAW, "labelsTr")
IMG_DST  = os.path.join(RAW, "imagesTr_clean")
LBL_DST  = os.path.join(RAW, "labelsTr_clean")


def is_case(fname):
    # real cases only: "<prefix>XX.nii.gz", not "._<prefix>XX.nii.gz" junk
    return fname.endswith(".nii.gz") and not fname.startswith(".")


def main():
    assert os.path.isdir(IMG_SRC), f"not found: {IMG_SRC} — download the task into ./raw first"
    assert os.path.isdir(LBL_SRC), f"not found: {LBL_SRC}"
    os.makedirs(IMG_DST, exist_ok=True)
    os.makedirs(LBL_DST, exist_ok=True)

    all_files = sorted(os.listdir(IMG_SRC))
    imgs = [f for f in all_files if is_case(f)]
    junk = len(all_files) - len(imgs)
    print(f"[{CFG['name']}] imagesTr entries={len(all_files)}  cases={len(imgs)}  junk filtered={junk}")

    n_link, missing_lbl = 0, []
    for f in imgs:
        if not os.path.exists(os.path.join(LBL_SRC, f)):
            missing_lbl.append(f)
            continue
        for src, dst in [(IMG_SRC, IMG_DST), (LBL_SRC, LBL_DST)]:
            s, d = os.path.join(src, f), os.path.join(dst, f)
            if not os.path.islink(d) and not os.path.exists(d):
                os.symlink(s, d)
        n_link += 1
    print(f"linked {n_link} image/label pairs into imagesTr_clean / labelsTr_clean")
    if missing_lbl:
        print(f"WARNING: {len(missing_lbl)} images have no label, skipped: {missing_lbl[:5]}")
    n_tr = int(round(n_link * 0.7))
    n_va = int(round(n_link * 0.1))
    print(f"split preview (seed=42, 70/10/20): ~{n_tr} train / ~{n_va} val / ~{n_link - n_tr - n_va} test")


if __name__ == "__main__":
    main()
