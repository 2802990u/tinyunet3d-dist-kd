# train_student_brats_full.py
# FULL-DATA, PAPER-GRADE student run on BraTS2023-GLI.
#   Student = TinyUNet3D (~0.26M, in=4/out=4).  Teacher = REPRODUCED SegMamba
#   (139M), trained from scratch on our own 876-case train split with model
#   selection on our own 125-case val -- NOT the officially released checkpoint,
#   which was found to leak into our 250-case test set (it scored 93.29% on our
#   "held out" test, higher than its own paper's published 91.60%, meaning it
#   had already seen many of these cases during its original training).
#   KD_WEIGHT (env) toggles: 0.0 = no-KD baseline ; >0 = SSM distillation.
#   SEED  (env) = training seed (model init + augmentation) -> run 3 seeds for mean+/-std.
#   Rigor: FIXED split shared with the teacher's own reproduction -- 876 train /
#          125 val / 250 test, exact same case IDs as segv2_repro (see
#          fixed_split_loader.py). Model selection on VAL only.
#          Final test numbers come from eval_student_brats_test.py / eval_full_all.py
#          on the held-out test set.
# Run from SegMamba-V2/brats23/  (needs student_unet3d.py + models_segmamba).
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ----------------------------- config -----------------------------
data_dir     = os.environ.get("BRATS_DATA", "./data/brats_fullres/train")   # SCRATCH (full preprocessed BraTS)
env          = "pytorch"
num_gpus     = 1
device       = "cuda:0"

max_epoch    = 300
batch_size   = 2
val_every    = 10
patch_size   = [128, 128, 128]
augmentation = True
IN_CH, OUT_CH = 4, 4
N_FG = OUT_CH - 1

STUDENT_BASE = int(os.environ.get("STUDENT_BASE", "32"))
KD_T         = 2.0
# REPRODUCED teacher -- trained only on our 876 train cases, val-selected on our
# 125 val cases, NEVER on our 250 test cases. Local copy in this directory
# (copied from segv2_repro's scratch output) so training doesn't depend on
# scratch storage persisting for the whole run.
SSM_CKPT     = "./segmambav2_teacher_reproduced.pt"

# NOTE: the split is now FIXED (876/125/250, shared with the teacher's own
# reproduction) via fixed_split_loader.get_fixed_split_loader(). The old
# SPLIT_SEED/TRAIN_RATE/VAL_RATE/TEST_RATE random-shuffle split is retired --
# it could not guarantee the same case IDs as the teacher run-to-run.

RUN = "baseline" if KD_WEIGHT == 0.0 else f"kd{KD_WEIGHT:g}"
# NEW logdir root -- keeps these fixed-split runs from colliding with / overwriting
# the old random-split checkpoints under ./logs/, so both remain available to compare.
logdir = f"./logs_fixedsplit/brats_full_{RUN}_s{SEED}"
model_save_path = os.path.join(logdir, "model")


def filter_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    out = {}
    for k, v in sd.items():
        k = str(k)
        out[k[7:] if k.startswith("module") else k] = v
    return out


class StudentTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1,
                 logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus,
                         logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=4, overlap=0.5)
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = 12

        self.model = TinyUNet3D(in_chans=IN_CH, out_chans=OUT_CH, base=STUDENT_BASE)

        self.use_kd = KD_WEIGHT > 0.0
        self.teacher = None
        if self.use_kd:
            from models_segmamba.segmambav2 import SegMamba
            print(f"[brats-full KD] teacher = {SSM_CKPT}  (KD_WEIGHT={KD_WEIGHT}, SEED={SEED})")
            self.teacher = SegMamba(IN_CH, OUT_CH)
            self.teacher.load_state_dict(filter_state_dict(torch.load(SSM_CKPT, map_location="cpu")))
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        else:
            print(f"[brats-full baseline] no KD  (SEED={SEED})")
        self._teacher_on_device = False

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                         momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"
        self.loss_func = DiceCELoss(to_onehot_y=True, softmax=True)

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"][:, 0].long()
        return image, label

    @staticmethod
    def kd_loss(student_logits, teacher_logits, T):
        p_t = F.softmax(teacher_logits / T, dim=1)
        log_p_s = F.log_softmax(student_logits / T, dim=1)
        return -(p_t * log_p_s).sum(dim=1).mean() * (T * T)

    def training_step(self, batch):
        image, label = self.get_input(batch)
        student_logits = self.model(image)
        sup = self.loss_func(student_logits, label.unsqueeze(1))
        if self.use_kd:
            if not self._teacher_on_device:
                self.teacher = self.teacher.to(image.device); self._teacher_on_device = True
            with torch.no_grad():
                teacher_logits = self.teacher(image)
            kd = self.kd_loss(student_logits, teacher_logits, KD_T)
            loss = sup + KD_WEIGHT * kd
            self.log("train_kd", kd, step=self.global_step)
        else:
            loss = sup
        self.log("train_sup", sup, step=self.global_step)
        self.log("train_loss", loss, step=self.global_step)
        return loss

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            return np.array([dice(pred, gt), 50])
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        else:
            return np.array([0.0, 50])

    def validation_step(self, batch):
        image, label = self.get_input(batch)
        output = self.window_infer(image, self.model).argmax(dim=1).cpu().numpy()
        target = label.cpu().numpy()
        dices = []
        for i in range(1, OUT_CH):
            cal_dice, _ = self.cal_metric(target == i, output == i)
            dices.append(cal_dice)
        return dices

    def validation_end(self, val_outputs):
        dices = val_outputs
        dices_mean = [dices[i].mean() for i in range(N_FG)]
        mean_dice = sum(dices_mean) / N_FG
        self.log("mean_dice", mean_dice, step=self.epoch)
        print("per-class (val) Dice: " + " ".join(f"{i+1}:{dices_mean[i]:.4f}" for i in range(N_FG)))
        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,
                os.path.join(model_save_path, f"best_model_{mean_dice:.4f}.pt"),
                delete_symbol="best_model")
        save_new_model_and_delete_last(self.model,
            os.path.join(model_save_path, f"final_model_{mean_dice:.4f}.pt"),
            delete_symbol="final_model")
        print(f"val mean_dice is {mean_dice}")


if __name__ == "__main__":
    trainer = StudentTrainer(env_type=env, max_epochs=max_epoch, batch_size=batch_size,
                             device=device, logdir=logdir, val_every=val_every,
                             num_gpus=num_gpus, master_port=17771, training_script=__file__)
    train_ds, val_ds, _test_ds = get_fixed_split_loader(data_dir)
    n = sum(p.numel() for p in trainer.model.parameters())
    print(f"[brats-full RUN={RUN} SEED={SEED}] student params={n/1e6:.3f}M  "
          f"train={len(train_ds)} val={len(val_ds)} (test held out, fixed split)")
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
