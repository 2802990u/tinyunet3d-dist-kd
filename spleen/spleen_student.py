# spleen_student.py
# TinyUNet3D student on MSD Task09_Spleen, trained on the SAME data, split and
# input volumes as the AGILE-recipe SegMamba-V2 teacher (checkpoints_Task09_Spleen_segmambav2).
#
#   KD_MODE=none  -> supervised only (no KD)
#   KD_MODE=dist  -> supervised + DIST (Pearson inter + intra) to the frozen teacher
#
# MODES (env MODE):
#   diag   one real batch: reports where a NaN/inf first appears (teacher bf16/fp32, loss terms, grads)
#   train  (default)  train, select best_model.pt on the VAL split, final VAL check with TTA+PP
#   val    score a checkpoint on the VAL split  (use ARCH=teacher to sanity-check the teacher)
#   test   score a checkpoint on the TEST split. Refuses to run twice unless FORCE=1
#          (the test set is touched once per final checkpoint).
#
# Everything the teacher's own evaluation did is reproduced here: AGILE MSDDataset on the
# preprocessed cache (same normalisation + cropping), full-volume gaussian sliding window,
# sigmoid, 8-way mirror TTA, threshold 0.5, largest-component post-processing (min ratio 0.1),
# Dice = (2*inter+1e-5)/(denom+1e-5).
#
# ENV (defaults):
#   KD_MODE(none) SEED(1) KD_WEIGHT(1.0) EPOCHS(250) STEPS(250) BATCH(2) BASE(64)
#   PATCH("192 192 64") VAL_EVERY(10) NUM_WORKERS(8) TAG("") MODE(train) ARCH(student)
#   CKPT (default <out_dir>/best_model.pt)   FORCE(0)   TTA(1) PP(1)  [val/test modes]
#   T_OUT(1)  number of output channels of the teacher checkpoint (strict load fails loudly if wrong)
#   TASK_DIR  PP_DIR  SPLIT_DIR  TEACHER_CKPT  AGILE_DIR  MODELS_DIR   (paths, see below)
#
# Run from ~/CoPa-Distill/SegMamba-V2/spleen/ inside the `segmamba` conda env.
import os
import sys
import json
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HOME = os.path.expanduser("~")
AGILE_DIR = os.environ.get("AGILE_DIR", f"{HOME}/AGILEMamba")
MODELS_DIR = os.environ.get("MODELS_DIR", f"{HOME}/CoPa-Distill/SegMamba-V2/amos_ct")
for _p in (MODELS_DIR, AGILE_DIR):            # AGILE_DIR ends up first on sys.path
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASK_DIR = os.environ.get("TASK_DIR", "./data/MSD/Task09_Spleen")
PP_DIR = os.environ.get("PP_DIR", "./data/MSD/Task09_Spleen_pp")
SPLIT_DIR = os.environ.get("SPLIT_DIR", f"{HOME}/CoPa-Distill/SegMamba-V2/spleen/teacher")
TEACHER_CKPT = os.environ.get("TEACHER_CKPT", f"{SPLIT_DIR}/best_model.pt")

KD_MODE = os.environ.get("KD_MODE", "none").lower()
assert KD_MODE in ("none", "dist"), "KD_MODE must be none | dist"
KD_WEIGHT = float(os.environ.get("KD_WEIGHT", "1.0"))
SEED = int(os.environ.get("SEED", "1"))
EPOCHS = int(os.environ.get("EPOCHS", "250"))
STEPS = int(os.environ.get("STEPS", "250"))
BATCH = int(os.environ.get("BATCH", "2"))
BASE = int(os.environ.get("BASE", "64"))
PATCH = tuple(int(x) for x in os.environ.get("PATCH", "192 192 64").split())
VAL_EVERY = int(os.environ.get("VAL_EVERY", "10"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
TAG = os.environ.get("TAG", "")
MODE = os.environ.get("MODE", "train").lower()
ARCH = os.environ.get("ARCH", "student").lower()
T_OUT = int(os.environ.get("T_OUT", "1"))
FORCE = os.environ.get("FORCE", "0") == "1"
TTA = os.environ.get("TTA", "1") == "1"
PP = os.environ.get("PP", "1") == "1"
IN_CH, N_FG = 1, 1

if ARCH == "teacher":
    OUT_DIR = f"./runs/teacher{TAG}"
else:
    OUT_DIR = f"./runs/student_{KD_MODE}_b{BASE}_s{SEED}{TAG}"
CKPT = os.environ.get("CKPT", ARCH == "teacher" and TEACHER_CKPT or f"{OUT_DIR}/best_model.pt")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLIPS8 = [None, (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]


# ----------------------------------------------------------------- data
def read_ids(name):
    with open(os.path.join(SPLIT_DIR, f"{name}_ids.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def load_split():
    tr, va, te = read_ids("train"), read_ids("val"), read_ids("test")
    s_tr, s_va, s_te = set(tr), set(va), set(te)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "split overlap"
    for i in tr + va + te:
        assert os.path.exists(os.path.join(PP_DIR, f"{i}.npy")), f"missing preprocessed case {i}"
    return tr, va, te


def build_ds(ids, train):
    from msd_dataset import MSDDataset, read_dataset_json
    meta = read_dataset_json(TASK_DIR)
    assert meta["n_modalities"] == IN_CH and meta["n_foreground_classes"] == N_FG, meta
    return MSDDataset(TASK_DIR, case_ids=ids, patch_size=PATCH, train=train,
                      full_volume=not train, meta=meta, preprocessed_dir=PP_DIR)


# ----------------------------------------------------------------- models
def filter_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    return {(k[7:] if str(k).startswith("module.") else str(k)): v for k, v in sd.items()}


def build_teacher():
    from models_segmamba.segmambav2 import SegMamba
    m = SegMamba(IN_CH, T_OUT)
    sd = filter_state_dict(torch.load(TEACHER_CKPT, map_location="cpu"))
    m.load_state_dict(sd, strict=True)         # fails loudly on any mismatch
    return m


def build_student():
    from student_unet3d import TinyUNet3D
    return TinyUNet3D(in_chans=IN_CH, out_chans=N_FG, base=BASE)


def to_logit(out, n_out):
    """Model output -> single foreground logit (B,1,D,H,W). sigmoid(logit) = P(fg)."""
    if isinstance(out, (tuple, list)):
        out = out[0]
    if n_out == 1:
        return out[:, 0:1]
    return out[:, 1:2] - out[:, 0:1]            # 2-channel softmax head -> equivalent logit


# ----------------------------------------------------------------- inference / metric
def dice_score(pred, target, eps=1e-5):
    dims = (0, 2, 3, 4)
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    return (2 * inter + eps) / (denom + eps)


@torch.no_grad()
def predict_probs(model, image, n_out, tta, sw_batch):
    from monai.inferers import sliding_window_inference
    flips = FLIPS8 if tta else [None]
    acc = None
    for f in flips:
        v = torch.flip(image, dims=f) if f else image
        logits = sliding_window_inference(v, roi_size=PATCH, sw_batch_size=sw_batch,
                                          predictor=lambda x: to_logit(model(x), n_out),
                                          overlap=0.5, mode="gaussian")
        p = torch.sigmoid(logits)
        if f:
            p = torch.flip(p, dims=f)
        acc = p if acc is None else acc + p
    return acc / len(flips)


@torch.no_grad()
def evaluate(model, n_out, ids, tta, pp, sw_batch, verbose=True):
    from postprocessing import postprocess_regions
    ds = build_ds(ids, train=False)
    model.eval()
    per_case = {}
    for k, cid in enumerate(ids):
        s = ds[k]
        image = s["image"].unsqueeze(0).to(DEV)
        target = s["target"].unsqueeze(0).to(DEV)
        probs = predict_probs(model, image, n_out, tta, sw_batch)
        pred = postprocess_regions(probs, 0.5, 0.1) if pp else (probs > 0.5).float()
        d = float(dice_score(pred, target).mean().cpu())
        per_case[cid] = d
        if verbose:
            print(f"  case {k+1}/{len(ids)} ({cid})  dice {d:.4f}", flush=True)
        del image, target, probs, pred
        torch.cuda.empty_cache()
    return per_case


def summarise(per_case):
    v = np.array(list(per_case.values()))
    return dict(mean=float(v.mean()), std=float(v.std()), min=float(v.min()),
                median=float(np.median(v)), max=float(v.max()), n=int(len(v)))


# ----------------------------------------------------------------- DIST (binary form)
def _pearson(a, b, dim, eps=1e-6):
    a = a - a.mean(dim=dim, keepdim=True)
    b = b - b.mean(dim=dim, keepdim=True)
    # eps inside the sqrt: d sqrt(x)/dx is infinite at x=0, which turns a zero-variance input
    # (e.g. a perfectly confident teacher over foreground voxels) into NaN gradients.
    return (a * b).sum(dim=dim) / ((a.pow(2).sum(dim=dim) + eps).sqrt() * (b.pow(2).sum(dim=dim) + eps).sqrt())


def dist_kd(s_logit, t_logit, label, beta=1.0, gamma=1.0, tau=1.0):
    """DIST for a sigmoid (binary) head. softmax([0, z]) == [1-sigmoid(z), sigmoid(z)], so the
    one-logit head is lifted to the equivalent 2-class distribution [bg, fg]; then the same
    foreground-masked inter + intra Pearson loss as the BraTS runs.
    NOTE: with only two classes the inter-class correlation of a pixel is +-1, so the INTER term is
    (almost) a disagreement indicator with ~zero gradient; the INTRA term (per-class correlation
    across foreground voxels) carries the signal."""
    B = s_logit.shape[0]
    zs = torch.cat([torch.zeros_like(s_logit), s_logit], dim=1).float().reshape(B, 2, -1)
    zt = torch.cat([torch.zeros_like(t_logit), t_logit], dim=1).float().reshape(B, 2, -1)
    s, t = F.softmax(zs / tau, dim=1), F.softmax(zt / tau, dim=1)
    inter = (1.0 - _pearson(s, t, dim=1)).mean()
    fg = label.reshape(B, -1) > 0.5
    terms = []
    for b in range(B):
        m = fg[b]
        if m.sum() < 2:
            continue
        terms.append((1.0 - _pearson(s[b][:, m], t[b][:, m], dim=1)).mean())
    intra = torch.stack(terms).mean() if terms else inter.new_zeros(())
    return beta * inter + gamma * intra, inter.detach(), intra.detach()


# ----------------------------------------------------------------- train
def train():
    from monai.losses import DiceCELoss
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(OUT_DIR, exist_ok=True)

    tr, va, te = load_split()
    for name, ids in (("train", tr), ("val", va), ("test", te)):
        with open(os.path.join(OUT_DIR, f"{name}_ids.txt"), "w") as f:
            f.write("\n".join(ids) + "\n")
    print(f"[split] train {len(tr)}  val {len(va)}  test {len(te)} (held out, not read in training)", flush=True)

    model = build_student().to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[student] TinyUNet3D base={BASE} params={n_par/1e6:.3f}M  patch={PATCH}  KD_MODE={KD_MODE} "
          f"KD_WEIGHT={KD_WEIGHT if KD_MODE != 'none' else 0}  seed={SEED}  epochs={EPOCHS}x{STEPS}", flush=True)

    teacher = None
    if KD_MODE != "none":
        teacher = build_teacher().to(DEV).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[teacher] loaded {TEACHER_CKPT}  params={sum(p.numel() for p in teacher.parameters())/1e6:.1f}M "
              f"(strict)", flush=True)

    train_ds = build_ds(tr, train=True)
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
                        drop_last=True, pin_memory=True, persistent_workers=NUM_WORKERS > 0)

    def batches():
        while True:
            for b in loader:
                yield b
    it = batches()

    total = EPOCHS * STEPS
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: max(0.0, 1.0 - i / total) ** 0.9)
    sup_loss = DiceCELoss(sigmoid=True)          # soft Dice + BCE on the single foreground logit

    best, hist = -1.0, []
    t_start = time.time()
    for ep in range(EPOCHS):
        model.train()
        run = dict(loss=0.0, sup=0.0, kd=0.0)
        n_ok, skipped = 0, 0
        for _ in range(STEPS):
            b = next(it)
            x = b["image"].to(DEV, non_blocking=True)
            y = b["target"].to(DEV, non_blocking=True)
            z = model(x)
            sup = sup_loss(z, y)
            loss, kd = sup, torch.zeros((), device=DEV)
            if teacher is not None:
                with torch.no_grad(), torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
                    zt = to_logit(teacher(x), T_OUT).float()
                kd, _, _ = dist_kd(z, zt, y)
                loss = sup + KD_WEIGHT * kd
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                print(f"  WARNING non-finite loss (sup {float(sup.detach()):.4g} kd {float(kd.detach()):.4g}) "
                      f"x_finite={bool(torch.isfinite(x).all())} "
                      f"zt_finite={bool(teacher is None or torch.isfinite(zt).all())} - step skipped", flush=True)
                sched.step()
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            if not torch.isfinite(gn):
                skipped += 1
                print(f"  WARNING non-finite grad norm (loss finite: sup {float(sup.detach()):.4g} "
                      f"kd {float(kd.detach()):.4g}) - step skipped", flush=True)
                opt.zero_grad(set_to_none=True)
                sched.step()
                continue
            opt.step()
            sched.step()
            run["loss"] += loss.item(); run["sup"] += sup.item(); run["kd"] += float(kd.detach())
            n_ok += 1
        if n_ok == 0:
            raise SystemExit("every step in this epoch was non-finite - aborting")
        run = {k: v for k, v in run.items()}
        print(f"epoch {ep:04d}  loss {run['loss']/n_ok:.4f}  sup {run['sup']/n_ok:.4f}  "
              f"kd {run['kd']/n_ok:.4f}  skipped {skipped}  lr {sched.get_last_lr()[0]:.2e}  {(time.time()-t_start)/60:.1f} min",
              flush=True)

        if (ep + 1) % VAL_EVERY == 0 or ep == EPOCHS - 1:
            pc = evaluate(model, 1, va, tta=False, pp=False, sw_batch=2, verbose=False)
            m = float(np.mean(list(pc.values())))
            hist.append((ep + 1, m))
            print(f"  val Dice (no TTA/PP) {m:.4f}", flush=True)
            if m > best:
                best = m
                torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))
                print(f"  saved new best ({best:.4f})", flush=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "last_model.pt"))

    # Final VAL check with the reporting protocol (TTA + post-processing). Test is NOT touched.
    final = {}
    for name in ("best_model.pt", "last_model.pt"):
        model.load_state_dict(torch.load(os.path.join(OUT_DIR, name), map_location=DEV))
        pc = evaluate(model, 1, va, tta=True, pp=True, sw_batch=2, verbose=False)
        final[name] = summarise(pc)
        print(f"[final VAL, TTA+PP] {name}: {final[name]['mean']:.4f}", flush=True)
    with open(os.path.join(OUT_DIR, "val_summary.json"), "w") as f:
        json.dump(dict(history=hist, best_val_noTTA=best, final_val=final), f, indent=2)
    print("training done. Test split was NOT touched - run MODE=test once.", flush=True)



# ----------------------------------------------------------------- diag
def diag():
    """One real batch: where does a non-finite value first appear? (MODE=diag)"""
    from monai.losses import DiceCELoss
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    tr, _, _ = load_split()
    ds = build_ds(tr, train=True)
    b = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    b = next(iter(b))
    x, y = b["image"].to(DEV), b["target"].to(DEV)
    def st(name, t):
        t = t.detach().float()
        print(f"  {name:26s} finite={bool(torch.isfinite(t).all())}  min {t.min().item():.4g}  max {t.max().item():.4g}  "
              f"mean {t.mean().item():.4g}", flush=True)
    print("[diag] batch", tuple(x.shape), "fg fraction", float((y > 0.5).float().mean()))
    st("image", x); st("target", y)
    model = build_student().to(DEV)
    z = model(x); st("student logit (init)", z)
    teacher = build_teacher().to(DEV).eval()
    with torch.no_grad():
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=DEV.type == "cuda"):
            zt16 = to_logit(teacher(x), T_OUT).float()
        zt32 = to_logit(teacher(x), T_OUT).float()
    st("teacher logit bf16", zt16); st("teacher logit fp32", zt32)
    if torch.isfinite(zt16).all() and torch.isfinite(zt32).all():
        print(f"  teacher bf16-vs-fp32 max |diff| {(zt16 - zt32).abs().max().item():.4g}")
    sup = DiceCELoss(sigmoid=True)(z, y)
    for name, zt in (("bf16", zt16), ("fp32", zt32)):
        kd, inter, intra = dist_kd(z, zt, y)
        print(f"  [{name}] sup {sup.item():.4g}  kd {kd.item():.4g}  inter {inter.item():.4g}  intra {intra.item():.4g}")
        for nm, term in (("sup", sup), ("kd", kd)):
            model.zero_grad(set_to_none=True)
            term.backward(retain_graph=True)
            g = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None))
            print(f"     grad norm of {nm:3s} ({name} teacher): {g.item():.4g}  finite={bool(torch.isfinite(g))}")


# ----------------------------------------------------------------- val / test
def score():
    tr, va, te = load_split()
    ids = va if MODE == "val" else te
    out_json = os.path.join(os.path.dirname(CKPT) if ARCH != "teacher" else OUT_DIR,
                            f"{MODE}_{os.path.splitext(os.path.basename(CKPT))[0]}.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    if MODE == "test" and os.path.exists(out_json) and not FORCE:
        raise SystemExit(f"{out_json} exists - the test split is scored once per checkpoint "
                         f"(FORCE=1 to override and say so in the paper).")
    if ARCH == "teacher":
        model, n_out, swb = build_teacher().to(DEV), T_OUT, 1
    else:
        model, n_out, swb = build_student().to(DEV), 1, 2
        model.load_state_dict(filter_state_dict(torch.load(CKPT, map_location=DEV)))
    print(f"[{MODE}] arch={ARCH} ckpt={CKPT} cases={len(ids)} TTA={TTA} PP={PP}", flush=True)
    pc = evaluate(model, n_out, ids, tta=TTA, pp=PP, sw_batch=swb)
    s = summarise(pc)
    print(f"\n==== {MODE.upper()} RESULT  arch={ARCH}  mean Dice {s['mean']:.4f} ({s['mean']*100:.2f}%)  "
          f"std {s['std']:.4f}  min {s['min']:.4f}  max {s['max']:.4f}  n={s['n']} ====", flush=True)
    with open(out_json, "w") as f:
        json.dump(dict(split=MODE, arch=ARCH, ckpt=CKPT, tta=TTA, pp=PP, per_case=pc, **s), f, indent=2)
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    if MODE == "train":
        train()
    elif MODE == "diag":
        diag()
    elif MODE in ("val", "test"):
        score()
    else:
        raise SystemExit("MODE must be train | val | test | diag")
