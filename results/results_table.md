# Results (test sets, fixed splits; students = mean of 3 seeds)

| Dataset | Model | Params (M) | FLOPs (G) | Dice (%) |
|---|---|---:|---:|---:|
| MSD Spleen | SegMamba-V2 teacher | 138.8 | 1566 | 96.84 |
| MSD Spleen | TinyUNet3D, no KD | 0.956 | 235.7 | 95.86 |
| MSD Spleen | TinyUNet3D + DIST | 0.956 | 235.7 | 96.10 |
| MSD Hippocampus | SegMamba-V2 teacher | 138.8 | 87.0 | 88.68 |
| MSD Hippocampus | TinyUNet3D, no KD | 0.957 | 4.9 | 87.88 |
| MSD Hippocampus | TinyUNet3D + DIST | 0.957 | 4.9 | 87.59 |
| BraTS 2023 | SegMamba-V2 teacher (reproduced) | 138.8 | 3730.7 | 88.88 |
| BraTS 2023 | TinyUNet3D, no KD | 0.957 | 211.5 | 88.95 |
| BraTS 2023 | TinyUNet3D + DIST | 0.957 | 211.5 | 88.96 |

Student FLOPs = 2 x MACs at each dataset's training patch (`tools/compute_flops.py`).
The comparison with published methods is in the paper (Table 1) and `results_table.tex`.

## KD gain at full labels (Dice %)

| Dataset | No KD | + DIST | Gain |
|---|---:|---:|---:|
| MSD Spleen | 95.86 | 96.10 | +0.24 |
| MSD Hippocampus | 87.88 | 87.59 | -0.29 |
| BraTS 2023 (100%) | 88.95 | 88.96 | +0.01 |

## BraTS 2023 data-fraction sweep (test Dice, fraction)

| Training data | No KD | + DIST | DIST gain |
|---:|---:|---:|---:|
| 5% | 0.8494 | 0.8677 | +0.0183 |
| 10% | 0.8628 | 0.8794 | +0.0166 |
| 25% | 0.8822 | 0.8870 | +0.0048 |
| 50% | 0.8882 | 0.8945 | +0.0063 |
| 100% | 0.8895 | 0.8896 | +0.0001 |

100% = 876 training cases. Raw values in `frac_sweep_brats.csv`; figure: `python tools/plot_frac_sweep.py`.
