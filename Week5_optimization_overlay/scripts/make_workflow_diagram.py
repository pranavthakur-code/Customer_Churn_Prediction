"""Draw the experimental workflow used in the report (reports/fig_workflow.png)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, DARK, LIGHT = "#2F5496", "#ED7D31", "#404040", "#DCE6F4"
fig, ax = plt.subplots(figsize=(12, 3.9)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4.4)

def box(x, y, w, h, text, fc=LIGHT, ec=BLUE, tc="black", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal")

def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.3))

box(0.1, 1.5, 1.7, 1.0, "Full data\n3,000 rows", fc="white")
box(2.2, 2.45, 2.0, 0.9, "Sealed test set\n600 rows\n(open once, at the end)", fc=DARK, ec=DARK, tc="white")
box(2.2, 0.65, 2.0, 0.9, "Development set\n2,400 rows", fc=BLUE, ec=BLUE, tc="white", bold=True)
arrow(1.8, 2.0, 2.2, 2.9); arrow(1.8, 2.0, 2.2, 1.1)
box(4.7, 0.2, 2.5, 2.2, "1. CONTROLS\nsame folds, seeds,\nmetric, preprocessing,\nbudget for every method\n\n2. SEARCH\ngrid | random |\nTPE (Optuna, Hyperopt)\n\neach trial = 5-fold CV", fc=LIGHT)
arrow(4.2, 1.1, 4.7, 1.3)
box(4.7, 2.7, 2.5, 0.9, "Trial log (CSV)\nparams, score, sd, seconds", fc="white")
arrow(5.95, 2.4, 5.95, 2.7)
box(7.7, 1.45, 2.0, 1.2, "3. CHECK\nnested CV, fresh-fold\nre-scoring, 1-SE rule,\nearly stopping", fc=LIGHT)
arrow(7.2, 1.3, 7.7, 1.9)
box(7.7, 0.2, 2.0, 0.9, "Choose ONE model\n(on development\ndata only)", fc=ORANGE, ec=ORANGE, tc="white", bold=True)
arrow(8.7, 1.45, 8.7, 1.1)
box(10.2, 1.0, 1.7, 1.5, "4. FINAL TEST\nonce, with\nconfidence\nintervals", fc=DARK, ec=DARK, tc="white", bold=True)
arrow(9.7, 0.65, 10.2, 1.4)
ax.plot([3.2, 3.2, 11.05], [3.35, 4.0, 4.0], color=DARK, lw=1.3)
arrow(11.05, 4.0, 11.05, 2.5)
ax.text(7.2, 4.12, "sealed test set: never used for tuning or selection", fontsize=8, color=DARK, style="italic", ha="center")
plt.tight_layout(); plt.savefig("reports/fig_workflow.png", dpi=150); print("saved")
