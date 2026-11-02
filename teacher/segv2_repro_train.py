import os
import glob
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer

from light_training.dataloading.dataset import MedicalDataset
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from light_training.utils.files_helper import save_new_model_and_delete_last

# =============================================================================
# SegMamba-V2 reproduced on OUR 876 / 125 / 250 split.
#
# This is a near-verbatim copy of their brats23/3_train.py. Everything that
# defines the recipe is unchanged:
#
#   model      SegMamba(4, 4)
#   optimiser  SGD, lr 1e-2, weight_decay 3e-5, momentum 0.99, nesterov
#   schedule   poly
#   loss       nn.CrossEntropyLoss()          (on raw 4-class logits)
#   epochs     1000,  batch_size 2 per GPU,  val_every 2
#   patch      128^3,  augmentation True,  train_process 12
#   inference  SlidingWindowInferer, sw_batch_size 2, overlap 0.5
#   metric     per-RAW-CLASS dice over labels 1,2,3 (NOT WT/TC/ET regions)
#
# FOUR changes, all of them about *which data* and *which hardware*, none of
# them about the method:
#
#  1. DATA SPLIT. Theirs reads data/test_list.py. We read our own exported
#     876/125/250 lists so their model trains on exactly the cases AGILE-Mamba
#     trained on, and is tested on exactly the cases AGILE-Mamba was tested on.
#
#  2. VALIDATION SET -- the one deliberate protocol deviation.
#     Theirs ends with:
#         trainer.train(train_dataset=train_ds, val_dataset=test_ds)
#     i.e. the best checkpoint is chosen by peeking at the test set. If we
#     copied that here, their model would get to select on OUR 250 test cases,
#     which AGILE-Mamba never touched during development, and the comparison
#     would be rigged in their favour. So we validate on our held-out 125 and
#     leave the 250 completely untouched until the single final evaluation.
#     Note this is a *conservative* choice: it can only make their reproduced
#     number lower than it would otherwise be, never higher.
#
#  3. GPUS. Theirs hardcodes CUDA_VISIBLE_DEVICES="4,5,6,7" and num_gpus=4
#     (4x A100, effective batch 8). Our nodes expose 3x L40S, so we use
#     "0,1,2" and num_gpus=3 -> effective batch 6. This is the closest we can
#     get; single-GPU would be effective batch 2, three times further off.
#
#  4. PATHS. logdir and data_dir point at scratch, because the preprocessed
#     data (16 GB, plus the unpacked .npy that MedicalDataset creates on first
#     use) does not fit in the home quota.
#
# NOTE ON THE VALIDATION METRIC: their validation_step scores `output == i`
# for i in 1,2,3 -- that is raw classes (NCR / ED / ET), not the WT/TC/ET
# regions their Table V reports. That is why their released checkpoint is
# named final_model_0.8388.pt rather than something near 0.916. We keep it
# exactly as they wrote it, because checkpoint selection is part of the recipe
# being reproduced. Region-wise WT/TC/ET scoring happens separately, once, in
# the final test evaluation.
# =============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

WORK = "/path/to/scratch/segv2_repro"
data_dir = f"{WORK}/data/fullres/train"
split_dir = f"{WORK}/splits"

logdir = f"{WORK}/logs/segmambav2_ourssplit"

env = "DDP"
model_save_path = os.path.join(logdir, "model")
max_epoch = 1000
batch_size = 2
val_every = 2
num_gpus = 3
device = "cuda:0"
patch_size = [128, 128, 128]
augmentation = True


def read_ids(name):
    with open(os.path.join(split_dir, f"{name}.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def get_our_split_loaders(data_dir, split_dir):
    """Three-way version of their get_train_test_loader_from_test_list.

    Same mechanics as theirs (glob the .npz, match on the stem, wrap in
    MedicalDataset); it just partitions three ways instead of two, and the
    membership comes from our split files rather than their test_list.py.
    """
    all_paths = sorted(glob.glob(f"{data_dir}/*.npz"))
    by_id = {p.split("/")[-1].split(".")[0]: p for p in all_paths}
    print(f"preprocessed cases found: {len(by_id)}")

    out = {}
    for name in ("train_ids", "val_ids", "test_ids"):
        ids = read_ids(name)
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise RuntimeError(
                f"{len(missing)} cases in {name}.txt have no preprocessed "
                f".npz, e.g. {missing[:5]}")
        out[name] = [by_id[i] for i in ids]

    n_tr, n_va, n_te = (len(out["train_ids"]), len(out["val_ids"]),
                        len(out["test_ids"]))
    print(f"train {n_tr}  val {n_va}  test {n_te}")
    assert (n_tr, n_va, n_te) == (876, 125, 250), "split sizes are wrong"
    # Overlap would silently leak; check rather than trust.
    s_tr, s_va, s_te = (set(out["train_ids"]), set(out["val_ids"]),
                        set(out["test_ids"]))
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), \
        "split overlap detected"

    train_ds = MedicalDataset(out["train_ids"])
    val_ds = MedicalDataset(out["val_ids"])
    # test_ds is built but NOT passed to the trainer -- it exists only so the
    # assertion above can verify all 250 are present and preprocessed.
    return train_ds, val_ds


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1,
                 num_gpus=1, logdir="./logs/", master_ip='localhost',
                 master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every,
                         num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=patch_size,
                                                 sw_batch_size=2,
                                                 overlap=0.5)
        self.patch_size = patch_size
        self.augmentation = augmentation
        self.train_process = 12

        from models_segmamba.segmambav2 import SegMamba
        self.model = SegMamba(4, 4)

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2,
                                         weight_decay=3e-5,
                                         momentum=0.99, nesterov=True)
        self.scheduler_type = "poly"

        self.loss_func = nn.CrossEntropyLoss()

    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3),
                  (labels == 1) | (labels == 3) | (labels == 2),
                  labels == 3]
        return torch.cat(result, dim=1).float()

    def training_step(self, batch):
        image, label = self.get_input(batch)

        pred = self.model(image)
        loss = self.loss_func(pred, label)

        self.log("train_loss", loss, step=self.global_step)
        return loss

    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]

        label = label[:, 0].long()

        return image, label

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])

        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])

        else:
            return np.array([0.0, 50])

    def validation_step(self, batch):
        image, label = self.get_input(batch)

        output = self.model(image).argmax(dim=1)
        output = output.cpu().numpy()
        target = label.cpu().numpy()

        dices = []

        c = 4
        for i in range(1, c):
            pred_c = output == i
            target_c = target == i

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)

        return dices

    def validation_end(self, val_outputs):
        dices = val_outputs

        dices_mean = []
        c = 3
        for i in range(0, c):
            dices_mean.append(dices[i].mean())

        mean_dice = sum(dices_mean) / len(dices_mean)

        self.log("0", dices_mean[0], step=self.epoch)
        self.log("1", dices_mean[1], step=self.epoch)
        self.log("2", dices_mean[2], step=self.epoch)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,
                                           os.path.join(model_save_path,
                                           f"best_model_{mean_dice:.4f}.pt"),
                                           delete_symbol="best_model")

        save_new_model_and_delete_last(self.model,
                                       os.path.join(model_save_path,
                                       f"final_model_{mean_dice:.4f}.pt"),
                                       delete_symbol="final_model")

        print(f"mean_dice is {mean_dice}", flush=True)


if __name__ == "__main__":

    trainer = BraTSTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           logdir=logdir,
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=17762,
                           training_script=__file__)

    train_ds, val_ds = get_our_split_loaders(data_dir, split_dir)

    # See note 2 at the top: val_dataset is OUR 125-case validation set, not
    # the test set. This is the only place the protocol differs from theirs.
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
