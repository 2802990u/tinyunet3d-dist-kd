# train_ssm_teacher_msd.py
# Stage 2: SSM teacher (SegMamba-V2) on an MSD small benchmark, trained from scratch
# (no released MSD checkpoints exist for this backbone).
#
# Recipe = the one that FIXED the AMOS teacher: nnU-Net SGD (lr=1e-2, momentum 0.99,
# nesterov, poly) + DiceCE. Do NOT switch to AdamW — it collapsed SegMamba on AMOS.
#
# Split: fixed seeded 70/10/20 (seed 42) — model selection on VAL, test untouched
# (report teacher test Dice later via eval_student_msd_test.py with ARCH=segmamba).
#
# Run from the task working dir (needs light_training + models_segmamba + msd_config):
#   TASK=spleen python train_ssm_teacher_msd.py
import os
import numpy as np
import torch
from light_training.dataloading.dataset import get_train_val_test_loader_from_train
from light_training.trainer import Trainer
from light_training.evaluation.metric import dice
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from models_segmamba.segmambav2 import SegMamba
from msd_config import get_task, SPLIT_SEED, TRAIN_RATE, VAL_RATE, TEST_RATE

CFG = get_task()

# ----------------------------- config -----------------------------
data_dir   = "./data/fullres/train"
env        = "pytorch"
num_gpus   = 1
device     = "cuda:0"

# NOTE: a light_training "epoch" is a FIXED 250 iterations (nnU-Net convention).
# Default 400 epochs = the AMOS SSM-teacher budget (SGD converges slowly).
# Env knobs: EPOCHS / BATCH_SIZE (11GB GPUs: use 1) / TRAIN_PROCESS / PRINT_TIME=1
max_epoch    = int(os.environ.get("EPOCHS", "400"))
batch_size   = int(os.environ.get("BATCH_SIZE", "2"))
val_every    = 25
patch_size   = CFG["patch_size"]
augmentation = True
IN_CH, OUT_CH = CFG["in_ch"], CFG["out_ch"]
N_FG = OUT_CH - 1

logdir = f"./logs/ssm_teacher_{CFG['name']}"
model_save_path = os.path.join(logdir, "model")


class SSMTeacherTrainer(Trainer):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.window_infer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=2, overlap=0.5)
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = int(os.environ.get("TRAIN_PROCESS", "12"))
        self.print_time = os.environ.get("PRINT_TIME", "0") == "1"

        self.model = SegMamba(IN_CH, OUT_CH)

        self.best_mean_dice = 0.0
        # SegMamba's native nnU-Net recipe (AdamW destabilises it — see AMOS teacher v1).
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=3e-5,
                                         momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"
        self.loss_func = DiceCELoss(to_onehot_y=True, softmax=True)

    def get_input(self, batch):
        return batch["data"], batch["seg"][:, 0].long()

    def training_step(self, batch):
        image, label = self.get_input(batch)
        pred = self.model(image)
        loss = self.loss_func(pred, label.unsqueeze(1))
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
            dices_mean = [np.asarray(val_outputs).mean()]
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
    trainer = SSMTeacherTrainer(env_type=env, max_epochs=max_epoch, batch_size=batch_size,
                                device=device, logdir=logdir, val_every=val_every,
                                num_gpus=num_gpus, master_port=17781, training_script=__file__)
    train_ds, val_ds, _test = get_train_val_test_loader_from_train(
        data_dir=data_dir, train_rate=TRAIN_RATE, val_rate=VAL_RATE, test_rate=TEST_RATE,
        seed=SPLIT_SEED)
    n = sum(p.numel() for p in trainer.model.parameters())
    print(f"[SSM-teacher {CFG['name']}] params={n/1e6:.2f}M  in={IN_CH} out={OUT_CH}  "
          f"patch={patch_size}  train={len(train_ds)} val={len(val_ds)} (test held out)")
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
