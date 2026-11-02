# train_student_msd.py
# The paper's small-benchmark trainer: TinyUNet3D student on an MSD task,
# baseline vs single-teacher (SegMamba) KD, fixed split, multi-seed.
#
# ENV:
#   TASK      = spleen | heart | prostate       (default spleen)
#   KD_MODE   = none | kl | dist                (default kl; none = no-KD baseline)
#   KD_WEIGHT = distillation weight             (default 1.0; ignored for none)
#   SEED      = training seed                   (default 42; run >=3 seeds)
#   STUDENT_BASE = 32 | 48 | 64                 (default 32; capacity axis —
#                  0.26M / 0.56M / 0.96M params, all under the 1M claim)
#
# Same recipe as every other student in the project: SGD nnU-Net + DiceCE supervised;
# KD = temperature-scaled KL (T=2) to the frozen SegMamba teacher's logits, or the
# foreground-masked DIST variant (Pearson inter+intra) from the BraTS frac study.
# Split: fixed seeded 70/10/20 (seed 42) — identical for every method/seed;
# select on VAL; report on held-out test via eval_student_msd_test.py.
#
# Run from the task working dir (needs student_unet3d.py + models_segmamba + msd_config):
#   KD_MODE=none SEED=1 python train_student_msd.py     # baseline
#   KD_MODE=kl   SEED=1 python train_student_msd.py     # KD
import os, glob, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from msd_config import get_task, SPLIT_SEED, TRAIN_RATE, VAL_RATE, TEST_RATE

CFG       = get_task()
KD_MODE   = os.environ.get("KD_MODE", "kl").lower()        # none | kl | dist
KD_WEIGHT = float(os.environ.get("KD_WEIGHT", "1.0"))
SEED      = int(os.environ.get("SEED", "42"))
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.trainer import Trainer
from light_training.evaluation.metric import dice
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from student_unet3d import TinyUNet3D

# ----------------------------- config -----------------------------
data_dir     = "./data/fullres/train"
env, num_gpus, device = "pytorch", 1, "cuda:0"
# NOTE: a light_training "epoch" is a FIXED 250 iterations (nnU-Net convention),
# independent of dataset size — budget epochs like AMOS (250), not per-pass.
# Local-run knobs (all env):
#   EPOCHS        training epochs (default 250 = 62.5k steps, the AMOS student budget)
#   BATCH_SIZE    default 2 (drop to 1 if OOM on small GPUs)
#   TRAIN_PROCESS augmentation worker processes (default 12; raise if 'get batch time'
#                 dominates and CPU has headroom)
#   PRINT_TIME=1  per-iteration timing: dataloader vs forward vs backward
max_epoch  = int(os.environ.get("EPOCHS", "250"))
batch_size = int(os.environ.get("BATCH_SIZE", "2"))
val_every  = 25
patch_size   = CFG["patch_size"]
augmentation = True
IN_CH, OUT_CH = CFG["in_ch"], CFG["out_ch"]
N_FG = OUT_CH - 1
STUDENT_BASE = int(os.environ.get("STUDENT_BASE", "32"))
SSM_CKPT_DIR = f"./logs/ssm_teacher_{CFG['name']}/model"

KL_T = 2.0
DIST_T, DIST_BETA, DIST_GAMMA = 1.0, 1.0, 1.0

USE_KD = (KD_MODE in ("kl", "dist")) and KD_WEIGHT > 0.0
RUN = "baseline" if not USE_KD else f"{KD_MODE}{KD_WEIGHT:g}"
if STUDENT_BASE != 32:
    RUN += f"_b{STUDENT_BASE}"          # keeps existing base-32 run names unchanged
logdir = f"./logs/student_{CFG['name']}_{RUN}_s{SEED}"
model_save_path = os.path.join(logdir, "model")


def filter_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    return {(k[7:] if str(k).startswith("module") else str(k)): v for k, v in sd.items()}


def find_best_ckpt(d):
    cks = glob.glob(os.path.join(d, "best_model_*.pt"))
    if not cks:
        raise FileNotFoundError(f"No best_model_*.pt in {d} — train the SSM teacher first "
                                f"(TASK={CFG['name']} python train_ssm_teacher_msd.py).")
    return max(cks, key=lambda p: float(p.split("best_model_")[-1].replace(".pt", "")))


class StudentTrainer(Trainer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.window_infer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=4, overlap=0.5)
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = int(os.environ.get("TRAIN_PROCESS", "12"))
        self.print_time = os.environ.get("PRINT_TIME", "0") == "1"
        self.model = TinyUNet3D(in_chans=IN_CH, out_chans=OUT_CH, base=STUDENT_BASE)

        self.teacher = None
        if USE_KD:
            from models_segmamba.segmambav2 import SegMamba
            ssm_ckpt = find_best_ckpt(SSM_CKPT_DIR)
            print(f"[student-{CFG['name']} {RUN} s{SEED}] teacher: {ssm_ckpt}  (KD_WEIGHT={KD_WEIGHT})")
            self.teacher = SegMamba(IN_CH, OUT_CH)
            self.teacher.load_state_dict(filter_state_dict(torch.load(ssm_ckpt, map_location="cpu")))
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
        else:
            print(f"[student-{CFG['name']} baseline s{SEED}] no KD")
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
        # foreground-masked DIST (same as the BraTS frac trainer)
        B, C = s_logits.shape[0], s_logits.shape[1]
        s = F.softmax(s_logits.reshape(B, C, -1) / DIST_T, dim=1)
        t = F.softmax(t_logits.reshape(B, C, -1) / DIST_T, dim=1)
        inter = (1.0 - self._pearson(s, t, dim=1)).mean()
        fg = (label.reshape(B, -1) > 0)
        intra_terms = []
        for b in range(B):
            m = fg[b]
            if m.sum() < 2:
                continue
            corr = self._pearson(s[b][:, m], t[b][:, m], dim=1)
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
        self.log("train_sup", sup, step=self.global_step)
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
        if N_FG == 1:
            dices_mean = [float(np.asarray(val_outputs).mean())]
        else:
            dices_mean = [val_outputs[i].mean() for i in range(N_FG)]
        mean_dice = sum(dices_mean) / N_FG
        self.log("mean_dice", mean_dice, step=self.epoch)
        print("per-class (val): " + " ".join(
            f"{CFG['class_names'][i]}:{dices_mean[i]:.4f}" for i in range(N_FG)))
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
                             num_gpus=num_gpus, master_port=17782, training_script=__file__)
    train_ds, val_ds, _test = get_train_val_test_loader_from_train(
        data_dir=data_dir, train_rate=TRAIN_RATE, val_rate=VAL_RATE, test_rate=TEST_RATE,
        seed=SPLIT_SEED)
    n = sum(p.numel() for p in trainer.model.parameters())
    print(f"[student {CFG['name']} RUN={RUN} SEED={SEED}] params={n/1e6:.3f}M  in={IN_CH} "
          f"out={OUT_CH}  patch={patch_size}  train={len(train_ds)} val={len(val_ds)} (test held out)")
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
