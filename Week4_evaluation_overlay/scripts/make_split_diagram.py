"""Draw the data-splitting strategies used in the report (reports/fig_splits.png)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BLUE, ORANGE, GREY, DARK = "#2F5496", "#ED7D31", "#D9D9D9", "#404040"
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6),
                        gridspec_kw={"width_ratios": [1, 1.25, 1.25]})

def bar(ax, y, x0, w, col, h=.62, txt=None):
    ax.add_patch(Rectangle((x0, y), w, h, fc=col, ec="white", lw=1.5))
    if txt:
        ax.text(x0 + w / 2, y + h / 2, txt, ha="center", va="center",
                color="white", fontsize=8, fontweight="bold")

# (a) hold-out
ax = axs[0]; ax.set_title("(a) Train / validation / test", fontsize=10)
bar(ax, 1.6, 0, 6, BLUE, txt="train"); bar(ax, 1.6, 6, 2, ORANGE, txt="val")
bar(ax, 1.6, 8, 2, DARK, txt="test")
ax.text(0, .8, "60 / 20 / 20 split. Fit on train, tune on\nvalidation, look at test ONCE at the end.", fontsize=8)
ax.set_xlim(0, 10); ax.set_ylim(0, 2.6); ax.axis("off")

# (b) stratified 5-fold
ax = axs[1]; ax.set_title("(b) Stratified 5-fold cross-validation", fontsize=10)
for k in range(5):
    for j in range(5):
        bar(ax, 4 - k, j * 1.6, 1.6, ORANGE if j == k else BLUE,
            h=.7, txt="val" if j == k else None)
    ax.text(-.2, 4 - k + .35, f"fold {k+1}", ha="right", va="center", fontsize=8)
ax.text(0, -.55, "Each fold keeps the same churn rate; the 5 scores are averaged.", fontsize=8)
ax.set_xlim(-1.6, 8); ax.set_ylim(-.9, 5); ax.axis("off")

# (c) time series split
ax = axs[2]; ax.set_title("(c) Time-series split (expanding window)", fontsize=10)
for k in range(4):
    bar(ax, 3 - k, 0, 1.6 * (k + 1), BLUE, h=.7)
    bar(ax, 3 - k, 1.6 * (k + 1), 1.6, ORANGE, h=.7, txt="val")
    bar(ax, 3 - k, 1.6 * (k + 2), 8 - 1.6 * (k + 2), GREY, h=.7)
    ax.text(-.2, 3 - k + .35, f"split {k+1}", ha="right", va="center", fontsize=8)
ax.text(0, -.55, "Train only on the past, validate on the future. Time runs left to right.", fontsize=8)
ax.set_xlim(-1.6, 8); ax.set_ylim(-.9, 4); ax.axis("off")
plt.tight_layout(); plt.savefig("reports/fig_splits.png", dpi=150)
print("saved")
