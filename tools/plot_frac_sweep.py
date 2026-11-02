# plot_frac_sweep_final.py
# Same two-panel design as fig_small (No KD vs DIST only), with FINAL-protocol values:
# BraTS 2023, fixed 876/125/250 split, region-wise WT/TC/ET mean Dice, mirror TTA, mean of 3 seeds.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fractions = [5, 10, 25, 50, 100]                              # % of BraTS training data
baseline  = [0.8494, 0.8628, 0.8822, 0.8882, 0.8895]          # No-KD mean Dice
dist_gain = [0.0183, 0.0166, 0.0048, 0.0063, 0.0001]          # DIST gain (paired by seed)
dist = [b + g for b, g in zip(baseline, dist_gain)]

INK, INK2, SURFACE, AXIS = "#0b0b0b", "#52514e", "#fcfcfb", "#c3c2b7"
C_BASE, C_DIST = "#e0693e", "#2f9e58"
plt.rcParams.update({"font.size": 13, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.78, 3.87), facecolor=SURFACE)
x = list(range(len(fractions)))
xt = [f"{f}%" for f in fractions]


def style_axis(ax, title):
    ax.set_title(title, fontsize=13, color=INK, pad=10)
    ax.set_xticks(x); ax.set_xticklabels(xt)
    ax.set_xlabel("Training data used (% of BraTS 2023)")
    ax.grid(True, axis="y", color=AXIS, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


style_axis(ax1, "Accuracy vs. training data size")
ax1.set_ylabel("Test Dice")
for ys, color, label, off in [(baseline, C_BASE, "No KD", -17), (dist, C_DIST, "+ KD (DIST)", 10)]:
    ax1.plot(x, ys, marker="o", markersize=8, linewidth=2.5, color=color, label=label)
    for xi, yi in zip(x, ys):
        ax1.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, off),
                     ha="center", fontsize=9.5, color=color)
ax1.set_ylim(0.838, 0.903)
ax1.legend(frameon=False, loc="lower right")

style_axis(ax2, "KD gain is largest with few labels, ~0 at full data")
ax2.set_ylabel("KD gain over baseline (test Dice)")
ax2.axhline(0, color=INK2, linewidth=1)
ax2.plot(x, dist_gain, marker="o", markersize=8, linewidth=2.5, color=C_DIST, label="DIST gain")
for xi, yi in zip(x, dist_gain):
    ax2.annotate(f"{yi:+.4f}", (xi, yi), textcoords="offset points", xytext=(0, -19 if xi == 2 else 10),
                 ha="center", fontsize=9.5, color=C_DIST)
ax2.set_ylim(-0.002, 0.021)
ax1.set_xlim(-0.4, 4.4); ax2.set_xlim(-0.4, 4.4)
ax2.legend(frameon=False, loc="upper right")

fig.tight_layout()
for ext, kw in (("png", dict(dpi=300)), ("pdf", {})):
    fig.savefig(f"fig_small_final.{ext}", bbox_inches="tight", facecolor=SURFACE, **kw)
print("ok")
