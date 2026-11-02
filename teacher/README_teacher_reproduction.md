# SegMamba-V2 reproduction kit

Everything needed to train and evaluate SegMamba-V2 on your own data splits,
including the fixes that were required to make the released code run on a
different split and report region-wise Dice.

The upstream paper is *SegMamba-V2: Long-Range Sequential Modeling Mamba for
General 3-D Medical Image Segmentation*, IEEE TMI 45(1):4–15, Jan 2026
(DOI 10.1109/TMI.2025.3589797). It reports 91.60% mean Dice on BraTS2023.

---

## Layout

```
upstream/        the released repo (ge-xing/SegMamba-V2), .git stripped
  brats23/       BraTS2023 pipeline: rename, preprocess, train, predict, metrics
    light_training/    their trainer, dataloading, augmentation, evaluation
    models_segmamba/   segmambav2.py  <- the only model file they ship
    data/test_list.py  their 250-case test split
  colorectal/    their second pipeline
our_scripts/     additions written for a controlled reproduction
slurm/           SLURM job scripts (edit paths and account)
docs/            this file
```

---

## What the released code does that you must decide about

Three things in `upstream/brats23/3_train.py` matter more than anything else:

**1. It validates on the test set.**

```python
trainer.train(train_dataset=train_ds, val_dataset=test_ds)
```

The checkpoint is selected by scoring the test set every `val_every=2` epochs
across 1000 epochs — roughly 500 evaluations of the data being reported. If you
want a held-out number, you must carve a validation set out of the training
pool. `our_scripts/segv2_repro_train.py` does exactly that and is otherwise a
near-verbatim copy.

Measured cost of this difference, on BraTS2023: **0.54 Dice points** (combining
the extra training cases and the select-on-test effect).

**2. Its validation metric is per-raw-class, not per-region.**

`validation_step` scores `output == i` for i in 1,2,3 — the raw labels
NCR/ED/ET. The paper reports the *regions* WT/TC/ET, which are unions of those
labels and therefore score several points higher on the same prediction. This
is why their released checkpoint is named `final_model_0.8388.pt` rather than
something near 0.916. Nothing in their pipeline computes region Dice during
training; `our_scripts/segv2_repro_eval_test.py` does it at test time.

**3. GPUs are hardcoded.**

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
num_gpus = 4
```

Their recipe is 4×A100 with `batch_size=2` per GPU, i.e. effective batch 8.
Change both to match your hardware, and note effective batch is one of the few
recipe variables that cannot otherwise be controlled.

---

## The recipe (keep all of this fixed when reproducing)

```
model       SegMamba(4, 4)
optimiser   SGD, lr 1e-2, weight_decay 3e-5, momentum 0.99, nesterov
schedule    poly
loss        nn.CrossEntropyLoss()  on raw 4-class logits
epochs      1000, batch_size 2 per GPU, val_every 2
patch       128^3, augmentation on, train_process 12
inference   SlidingWindowInferer, sw_batch_size 2, overlap 0.5, mirror TTA
```

Note the published paper and the released code disagree in places — the paper
claims weight decay 1e-5 while the code uses 3e-5, and the paper claims elastic
deformation while the code sets `do_elastic_deform=False`. The code is what
produces the reported numbers, so follow the code.

---

## Environment

SegMamba-V2 works with the **stock** `mamba_ssm`.

SegMamba **V1** does not: it passes `bimamba_type` to `Mamba.__init__`, which
only exists in the forked mamba_ssm that ships with the Vision-Mamba lineage.
Installing that fork over a working environment will break V2 and anything else
depending on stock mamba_ssm — build a cloned conda env for V1 instead:

```bash
conda create --clone <your_env> -n segmamba_v1 -y
conda activate segmamba_v1
cd SegMamba/mamba && pip install -e .
```

Other dependencies that are not in the repo's requirements but are needed by
the preprocessing: `batchgenerators`, `acvl_utils`, `monai`, `SimpleITK`,
`nibabel`.

---

## Running it

### 1. Preprocess

`upstream/brats23/1_rename_mri_data_BraTS2023.py` renames modality files **in
place**:

```
BraTS-GLI-00000-000-t2w.nii.gz  ->  t2w.nii.gz
```

Run it only on a **copy** of your data, never the original — it will break any
other loader pointed at that folder. It is also not restartable: a second run
crashes on already-renamed files, and on any non-directory entry inside the
data folder (stray `.pth` checkpoints are a common cause). `slurm/segv2_repro_prepare.slurm`
contains an idempotent replacement.

Output is ~13 MB/case for the preprocessed `fullres` data, plus the raw copy.
`MedicalDataset` additionally unpacks `.npz` → `.npy` on first use, roughly
doubling it, so budget accordingly and prefer scratch over a quota'd home.

### 2. Train

```bash
python our_scripts/segv2_repro_train.py
```

Edit the paths and GPU count at the top. It reads `splits/{train,val,test}_ids.txt`,
built by `our_scripts/segv2_repro_export_split.py`.

### 3. Evaluate

```bash
python our_scripts/segv2_repro_eval_test.py
```

Region-wise WT/TC/ET on the test set, with their sliding-window inference and
mirror TTA.

---

## Traps that cost real time

**Region order differs between codebases.**

```
theirs (convert_labels): [TC, WT, ET]
ours   (dice_score)    : [WT, TC, ET]
```

Getting this wrong silently swaps WT and TC. Both are high numbers, so the
result looks plausible. `segv2_repro_eval_test.py` reorders explicitly.

**`MedicalDataset` returns numpy memmaps, not tensors.** `batch["data"].to(device)`
fails with `'memmap' object has no attribute 'to'`. Use:

```python
image = torch.from_numpy(np.array(batch["data"])).float()
```

**Training "fails" at the end.** `batchgenerators` raises
`RuntimeError: One or more background workers are no longer alive` while
shutting down worker threads *after* training completes. Check the epoch count
before assuming the run died — a 1000-epoch run with `val_every=2` prints 500
`mean_dice is ...` lines.

**Their trainer has no resume-from-checkpoint path.** A requeue restarts from
zero. On 3×L40S, 1000 epochs over 876 cases took 13.5 hours, so this is usually
fine, but check against your wall-clock limit.

---

## Extending to other datasets

`2_preprocessing_BraTS2023.py` hardcodes the four-modality BraTS layout, so each
new dataset needs its own preprocessing script. `upstream/colorectal/` is the
only other one they ship and is the best template for a single-modality CT task.

The model itself is `SegMamba(in_chans, out_chans)`, so changing modality and
class counts is a constructor argument — the work is all in preprocessing and in
whatever metric convention the new dataset uses.
