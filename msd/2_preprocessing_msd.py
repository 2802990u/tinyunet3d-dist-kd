# 2_preprocessing_msd.py  (MSD small benchmarks)
# Stage 1b: run the light_training DefaultPreprocessor on a staged MSD task.
#   plan()          -> analyses dataset, writes ./data_analysis_result.txt
#                      (foreground intensity stats + fullres target spacing)
#   process_train() -> resamples + normalizes every case, writes
#                      ./data/fullres/train/<case>.npz + <case>.pkl properties.
#
# Same driver as amos_ct/2_preprocessing_ct.py, pointed at the MSD layout staged by
# stage_msd.py. data_type comes from msd_config (CT for spleen -> CTNormalization
# from foreground stats; MRI for heart/prostate -> verify the MRI branch of
# DefaultPreprocessor does per-image Z-score before trusting those tasks).
#
# Run from the task working dir (needs light_training importable there):
#   TASK=spleen python 2_preprocessing_msd.py
from light_training.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
import os, json
from msd_config import get_task

CFG        = get_task()
BASE       = CFG["raw_dir"]
IMAGE_DIR  = "imagesTr_clean"
LABEL_DIR  = "labelsTr_clean"
OUTPUT_DIR = "./data/fullres/train/"
ALL_LABELS = list(range(1, CFG["out_ch"]))       # fg labels 1..C-1
ANALYSIS   = "./data_analysis_result.txt"


def make_pre():
    return DefaultPreprocessor(base_dir=BASE, image_dir=IMAGE_DIR,
                               label_dir=LABEL_DIR, data_type=CFG["data_type"])


def plan():
    make_pre().run_plan()   # writes ANALYSIS


def process_train():
    pre = make_pre()
    with open(ANALYSIS, "r") as f:
        content = json.loads(f.read().strip("\n"))

    # foreground intensity stats, keyed "0" (single channel; prostate: verify 2-ch)
    fg = content["intensity_statistics_per_channel"]
    # run_plan stores fullres spacing as (x, y, z); the preprocessor consumes
    # out_spacing in (z, y, x) order (same convention as the AMOS driver).
    out_spacing = list(content["fullres spacing"])[::-1]

    print(f"[{CFG['name']}] median_shape:", content.get("median_shape"))
    print("using out_spacing (z,y,x):", out_spacing)
    print("all_labels:", ALL_LABELS)

    pre.run(output_spacing=out_spacing,
            output_dir=OUTPUT_DIR,
            all_labels=ALL_LABELS,
            foreground_intensity_properties_per_channel=fg,
            num_processes=8)


if __name__ == "__main__":
    if not os.path.exists(ANALYSIS):
        plan()
    else:
        print(f"{ANALYSIS} exists -> skipping plan(), going straight to preprocessing")
    process_train()
