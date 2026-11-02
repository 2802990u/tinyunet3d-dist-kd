# train_student_frac_brats.py
# Data-efficiency study on BraTS: baseline vs KL-KD vs DIST-KD, at a chosen fraction
# of the training set. FIXED 876/125/250 split, shared with the teacher's own
# reproduction (see fixed_split_loader.py); val/test identical across all runs;
# the training subset is nested and depends ONLY on DATA_FRAC (so all methods at a
# given fraction see the same cases). Model selection on val; report on held-out test
# via eval_student_brats_test.py / eval_frac_all.py.
#
# Teacher = REPRODUCED SegMamba (139M), trained from scratch on our own 876-case
# train split, selected on our 125-case val -- NOT the officially released
# checkpoint (that one leaked: 93.29% on our "held out" 250-case test set, higher
# than its own paper's published 91.60%, meaning it had already seen many of
# these cases during its original training).
#
# ENV:
#   KD_MODE   = none | kl | dist        (default dist)
#   DATA_FRAC = 1.0 / 0.5 / 0.25 / 0.1 / 0.05   (fraction of the 876 train cases)
#   SEED      = training seed
#   KD_WEIGHT = distillation weight (default 1.0; ignored for none)
#
# DIST here uses a FOREGROUND-MASKED intra-class term (correlation across foreground
# voxels only) to avoid the background-dominated noise that made unmasked DIST unstable.
# Run from SegMamba-V2/brats23/.
import os, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KD_MODE   = os.environ.get("KD_MODE", "dist").lower()      # none | kl | dist
DATA_FRAC = float(os.environ.get("DATA_FRAC", "1.0"))
SEED      = int(os.environ.get("SEED", "42"))
KD_WEIGHT = float(os.environ.get("KD_WEIGHT", "1.0"))
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

from fixed_split_loader import get_fixed_split_loader
from light_training.trainer import Trainer
from light_training.evaluation.metric import dice
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from student_unet3d import TinyUNet3D

data_dir     = os.environ.get("BRATS_DATA", "./data/brats_fullres/train")
env, num_gpus, device = "pytorch", 1, "cuda:0"
max_epoch, batch_size, val_every = 300, 2, 10
patch_size   = [128, 128, 128]
augmentation = True
IN_CH, OUT_CH, N_FG = 4, 4, 3
STUDENT_BASE = int(os.environ.get("STUDENT_BASE", "32"))
# REPRODUCED teacher -- see note at top of file. Local copy in this directory.
SSM_CKPT     = "./segmambav2_teacher_reproduced.pt"

KL_T = 2.0
DIST_T, DIST_BETA, DIST_GAMMA = 1.0, 1.0, 1.0
# NOTE: split is now FIXED (876/125/250, shared with the teacher) via
# fixed_split_loader.get_fixed_split_loader(). The subsample below draws the
# DATA_FRAC-sized training subset from the fixed 876-case train set (nested,
# run-seed-independent), not from a fresh random 80% shuffle.

USE_KD = (KD_MODE in ("kl", "dist")) and KD_WEIGHT > 0.0
RUN = f"{KD_MODE}_f{DATA_FRAC:g}"
logdir = f"./logs_fixedsplit/brats_frac_{RUN}_s{SEED}"
model_save_path = os.path.join(logdir, "model")

# Subsampling reference seed -- kept fixed and separate from SPLIT_SEED so the
# nested-fraction property holds regardless of which split-generation scheme
# is in use. Any fixed constant is fine here; keep it stable across runs.
SUBSAMPLE_SEED = 42


def filter_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    return {(k[7:] if str(k).startswith("module") else str(k)): v for k, v in sd.items()}


def subsample_train(train_ds, frac):
    if frac >= 1.0:
        return train_ds
    n = len(train_ds.datalist)
    order = list(range(n)); random.Random(SUBSAMPLE_SEED).shuffle(order)   # nested, run-seed-independent
    keep = set(order[:max(1, int(round(n * frac)))])
    train_ds.datalist   = [train_ds.datalist[i]   for i in range(n) if i in keep]
    train_ds.data_cached = [train_ds.data_cached[i] for i in range(n) if i in keep]
    return train_ds


class FracTrainer(Trainer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.window_infer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=4, overlap=0.5)
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = 6      # lighter: 12 workers over-subscribed 8 CPUs and died under concurrency
        self.model = TinyUNet3D(in_chans=IN_CH, out_chans=OUT_CH, base=STUDENT_BASE)

        self.teacher = None
        if USE_KD:
            from models_segmamba.segmambav2 import SegMamba
            print(f"[frac {RUN} s{SEED}] mode={KD_MODE} teacher={SSM_CKPT} w={KD_WEIGHT}")
            self.teacher = SegMamba(IN_CH, OUT_CH)
            self.teacher.load_state_dict(filter_state_dict(torch.load(SSM_CKPT, map_location="cpu")))
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        self._teacher_on_device = False

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                         momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"
        self.loss_func = DiceCELoss(to_onehot_y=True, softmax=True)

    def get_input(self, batch):
        return batch["data"], batch["seg"][:, 0].long()

    @staticmethod
    def _pearson(a, b, dim, eps=1e-6):
        a = a - a.mean(dim=dim, keepdim=True); b = b - b.mean(dim=dim, keepdim=True)
        return (a * b).sum(dim=dim) / (a.pow(2).sum(dim=dim).sqrt() * b.pow(2).sum(dim=dim).sqrt() + eps)

    def kl_kd(self, s_logits, t_logits):
        p_t = F.softmax(t_logits / KL_T, dim=1)
        log_p_s = F.log_softmax(s_logits / KL_T, dim=1)
        return -(p_t * log_p_s).sum(dim=1).mean() * (KL_T * KL_T)

    def dist_kd(self, s_logits, t_logits, label):
        B, C = s_logits.shape[0], s_logits.shape[1]
        s = F.softmax(s_logits.reshape(B, C, -1) / DIST_T, dim=1)   # (B,C,N)
        t = F.softmax(t_logits.reshape(B, C, -1) / DIST_T, dim=1)
        inter = (1.0 - self._pearson(s, t, dim=1)).mean()          # across classes, per voxel
        # intra-class across FOREGROUND voxels only (per volume), averaged
        fg = (label.reshape(B, -1) > 0)
        intra_terms = []
        for b in range(B):
            m = fg[b]
            if m.sum() < 2:
                continue
            corr = self._pearson(s[b][:, m], t[b][:, m], dim=1)    # (C,)
            intra_terms.append((1.0 - corr).mean())
        intra = torch.stack(intra_terms).mean() if intra_terms else inter.new_zeros(())
        return DIST_BETA * inter + DIST_GAMMA * intra

    def training_step(self, batch):
        image, label = self.get_input(batch)
        s_logits = self.model(image)
        sup = self.loss_func(s_logits, label.unsqueeze(1))
        if USE_KD:
            if not self._teacher_on_device:
                self.teacher = self.teacher.to(image.device); self._teacher_on_device = True
            with torch.no_grad():
                t_logits = self.teacher(image)
            kd = self.kl_kd(s_logits, t_logits) if KD_MODE == "kl" else self.dist_kd(s_logits, t_logits, label)
            loss = sup + KD_WEIGHT * kd
            self.log("train_kd", kd, step=self.global_step)
        else:
            loss = sup
        self.log("train_loss", loss, step=self.global_step)
        return loss

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            return np.array([dice(pred, gt), 50])
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        return np.array([0.0, 50])

    def validation_step(self, batch):
        image, label = self.get_input(batch)
        output = self.window_infer(image, self.model).argmax(dim=1).cpu().numpy()
        target = label.cpu().numpy()
        return [self.cal_metric(target == i, output == i)[0] for i in range(1, OUT_CH)]

    def validation_end(self, val_outputs):
        dm = [val_outputs[i].mean() for i in range(N_FG)]
        mean_dice = sum(dm) / N_FG
        self.log("mean_dice", mean_dice, step=self.epoch)
        print("per-class (val): " + " ".join(f"{i+1}:{dm[i]:.4f}" for i in range(N_FG)))
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,
                os.path.join(model_save_path, f"best_model_{mean_dice:.4f}.pt"), delete_symbol="best_model")
        save_new_model_and_delete_last(self.model,
            os.path.join(model_save_path, f"final_model_{mean_dice:.4f}.pt"), delete_symbol="final_model")
        print(f"val mean_dice is {mean_dice}")


if __name__ == "__main__":
    trainer = FracTrainer(env_type=env, max_epochs=max_epoch, batch_size=batch_size, device=device,
                          logdir=logdir, val_every=val_every, num_gpus=num_gpus,
                          master_port=17774, training_script=__file__)
    train_ds, val_ds, _test = get_fixed_split_loader(data_dir)
    train_ds = subsample_train(train_ds, DATA_FRAC)
    n = sum(p.numel() for p in trainer.model.parameters())
    print(f"[frac RUN={RUN} SEED={SEED}] params={n/1e6:.3f}M  mode={KD_MODE} frac={DATA_FRAC} "
          f"train={len(train_ds)} val={len(val_ds)} (test held out, fixed split)")
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
