# eval_frac_all.py
# Sweep-evaluate every finished data-fraction run on the held-out BraTS test split
# and write one tidy CSV — the input to plot_frac_curve.py.
#
# Scans ./logs_fixedsplit/brats_frac_{mode}_f{frac}_s{seed}/model/best_model_*.pt
# (the dirs train_student_frac_brats.py creates), evaluates each best checkpoint on
# the SAME fixed 876/125/250 split the teacher's segv2_repro uses (see
# fixed_split_loader.py), and appends a row per run to FRAC_CSV. Runs already
# present in the CSV are skipped, so it is safe to re-run as more fraction jobs finish.
#
# Run from SegMamba-V2/brats23/  (GPU):
#   python eval_frac_all.py
import os, re, glob, csv
import numpy as np
import torch
from fixed_split_loader import get_fixed_split_loader
from light_training.evaluation.metric import dice
from monai.inferers import SlidingWindowInferer
from student_unet3d import TinyUNet3D

data_dir   = os.environ.get("BRATS_DATA", "./data/brats_fullres/train")   # must match the trainer
patch_size = [128, 128, 128]
IN_CH, OUT_CH = 4, 4
STUDENT_BASE = 64
FRAC_CSV   = "./frac_results_fixedsplit.csv"
DEVICE     = "cuda:0"

RUN_RE = re.compile(r"brats_frac_(?P<mode>none|kl|dist)_f(?P<frac>[0-9.]+)_s(?P<seed>\d+)$")


def filter_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    return {(k[7:] if str(k).startswith("module") else str(k)): v for k, v in sd.items()}


def find_runs():
    runs = []
    for d in sorted(glob.glob("./logs_fixedsplit/brats_frac_*_s*")):
        m = RUN_RE.search(os.path.basename(d))
        if not m:
            continue
        cks = glob.glob(os.path.join(d, "model", "best_model_*.pt"))
        if not cks:
            print(f"  [skip] no best_model yet: {d}")
            continue
        best = max(cks, key=lambda p: float(p.split("best_model_")[-1].replace(".pt", "")))
        runs.append(dict(mode=m["mode"], frac=float(m["frac"]), seed=int(m["seed"]), ckpt=best))
    return runs


def load_done():
    done = set()
    if os.path.exists(FRAC_CSV):
        with open(FRAC_CSV) as f:
            for row in csv.DictReader(f):
                done.add((row["mode"], float(row["frac"]), int(row["seed"])))
    return done


def eval_ckpt(ckpt, test_ds, inferer):
    model = TinyUNet3D(in_chans=IN_CH, out_chans=OUT_CH, base=STUDENT_BASE)
    model.load_state_dict(filter_state_dict(torch.load(ckpt, map_location="cpu")))
    model = model.to(DEVICE).eval()
    per_case = []
    for i in range(len(test_ds)):
        it = test_ds[i]
        image = torch.as_tensor(np.asarray(it["data"])).unsqueeze(0).to(DEVICE).float()
        target = np.asarray(it["seg"])[0]
        with torch.no_grad():
            out = inferer(image, model).argmax(dim=1)[0].cpu().numpy()
        ds = []
        for c in range(1, OUT_CH):
            gt, pr = (target == c), (out == c)
            if pr.sum() > 0 and gt.sum() > 0:
                ds.append(dice(pr, gt))
            elif gt.sum() == 0 and pr.sum() == 0:
                ds.append(1.0)
            else:
                ds.append(0.0)
        per_case.append(float(np.mean(ds)))
    del model
    torch.cuda.empty_cache()
    return float(np.mean(per_case)), float(np.std(per_case)), len(per_case)


if __name__ == "__main__":
    _tr, _val, test_ds = get_fixed_split_loader(data_dir)
    n_train_full = len(_tr)
    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=4, overlap=0.5)

    runs, done = find_runs(), load_done()
    todo = [r for r in runs if (r["mode"], r["frac"], r["seed"]) not in done]
    print(f"found {len(runs)} finished runs; {len(done)} already in CSV; evaluating {len(todo)}")

    new_file = not os.path.exists(FRAC_CSV)
    with open(FRAC_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["mode", "frac", "n_train", "seed", "test_dice", "per_case_std", "n_test", "ckpt"])
        for r in todo:
            n_train = max(1, int(round(n_train_full * r["frac"])))
            print(f"[eval] {r['mode']} f={r['frac']:g} s={r['seed']}  ({r['ckpt']})")
            mean_d, std_d, n_test = eval_ckpt(r["ckpt"], test_ds, inferer)
            print(f"       test Dice {mean_d:.4f} (per-case std {std_d:.4f}, N={n_test})")
            w.writerow([r["mode"], f"{r['frac']:g}", n_train, r["seed"],
                        f"{mean_d:.6f}", f"{std_d:.6f}", n_test, r["ckpt"]])
            f.flush()
    print(f"\nwrote/updated {FRAC_CSV} — copy it local and run plot_frac_curve.py")
