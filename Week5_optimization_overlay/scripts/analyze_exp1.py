"""Summarise Experiment 1: tables and convergence figure from reports/trials_exp1.csv."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN, GREY = "#2F5496", "#ED7D31", "#548235", "#7F7F7F"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
OUT = Path("reports")
T = pd.read_csv(OUT / "trials_exp1.csv")
meta = json.load(open(OUT / "exp1_meta.json"))
DEF, LOG = meta["reference"]["hgb_default"][0], meta["reference"]["logistic_C0.1"][0]
NAMES = {"grid": "Grid search", "random": "Random search",
         "optuna_tpe": "Optuna (TPE)", "hyperopt_tpe": "Hyperopt (TPE)"}
overall_best = T.cv_auc.max()

print("### E1-A. Best CV ROC-AUC found with 81 evaluations (5 seeds for stochastic methods)")
rows = []
for m, g in T.groupby("method", sort=False):
    fin = g.groupby("seed").cv_auc.max()
    reach = []
    if m == "grid":      # visiting order is arbitrary, so average over 300 random orders
        gv_ = g.sort_values("trial").cv_auc.values; prng = np.random.default_rng(0)
        for _ in range(300):
            hit = np.where(np.maximum.accumulate(prng.permutation(gv_)) >= overall_best - 0.003)[0]
            reach.append(hit[0] + 1 if len(hit) else np.nan)
    else:
        for s, gs in g.groupby("seed"):
            hit = np.where(gs.sort_values("trial").best_so_far.values >= overall_best - 0.003)[0]
            reach.append(hit[0] + 1 if len(hit) else np.nan)
    rows.append({"method": NAMES[m], "runs": len(fin), "mean best": fin.mean(),
                 "sd": fin.std(ddof=1) if len(fin) > 1 else np.nan,
                 "worst run": fin.min(), "best run": fin.max(),
                 "trials to reach within 0.003 of overall best (median)": np.nanmedian(reach),
                 "runs reaching it": f"{int(np.sum(~np.isnan(reach)))}/{len(reach)}",
                 "sec/trial": g.seconds.mean(),
                 "% trials above defaults": 100 * (g.cv_auc > DEF).mean()})
tab = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(tab.round(4).to_string(index=False))
print(f"Overall best CV AUC seen in any trial: {overall_best:.4f}")
print(f"References: library-default gradient boosting {DEF:.4f}; untuned logistic regression (C=0.1) {LOG:.4f}")

print("\n### E1-B. Best configurations overall")
cols = ["method", "learning_rate", "max_depth", "min_samples_leaf", "l2_regularization", "cv_auc", "cv_std"]
print(T.sort_values("cv_auc", ascending=False).head(6)[cols].round(4).to_string(index=False))
imp = json.load(open(OUT / "optuna_importance_seed0.json"))
print("\nOptuna parameter importance (fANOVA, seed-0 study):",
      {k: round(v, 3) for k, v in imp.items()})
print("Total trials: %d, total evaluation time %.0f s (%.2f s per trial)"
      % (len(T), T.seconds.sum(), T.seconds.mean()))
print("Trials scoring below defaults (%.4f): %.1f%% of all" % (DEF, 100 * (T.cv_auc < DEF).mean()))
print("Trials scoring above untuned logistic (%.4f): %d of %d" % (LOG, (T.cv_auc > LOG).sum(), len(T)))


print("\n### E1-C. Why do the methods differ? Share of trials landing near the optimum")
for tol in (0.003, 0.001):
    print(f"Within {tol} of the overall best ({overall_best:.4f}):")
    for m, g in T.groupby("method", sort=False):
        print(f"  {NAMES[m]:<15} {100 * (g.cv_auc >= overall_best - tol).mean():5.1f}% of trials")
gg = T[T.method == "grid"]
print("Grid points with max_depth=2 and learning_rate=0.03: %d of %d (CV AUC %.4f-%.4f)"
      % (((gg.max_depth == 2) & (gg.learning_rate == 0.03)).sum(), len(gg),
         gg[(gg.max_depth == 2) & (gg.learning_rate == 0.03)].cv_auc.min(),
         gg[(gg.max_depth == 2) & (gg.learning_rate == 0.03)].cv_auc.max()))
print("Share of trials with max_depth=2:",
      {NAMES[m]: round(float((T[T.method == m].max_depth == 2).mean()), 2)
       for m in ["random", "optuna_tpe", "hyperopt_tpe"]})

# ---- figure: convergence + final scores
fig, ax = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw={"width_ratios": [1.5, 1]})
x = np.arange(1, 82)
cols_ = {"random": ORANGE, "optuna_tpe": BLUE, "hyperopt_tpe": GREEN}
for m, c in cols_.items():
    A = np.vstack([g.sort_values("trial").best_so_far.values for _, g in T[T.method == m].groupby("seed")])
    ax[0].plot(x, A.mean(0), color=c, lw=2, label=NAMES[m]); ax[0].fill_between(x, A.min(0), A.max(0), color=c, alpha=.15)
gv = T[T.method == "grid"].sort_values("trial").cv_auc.values
rng = np.random.default_rng(0)
G = np.vstack([np.maximum.accumulate(rng.permutation(gv)) for _ in range(300)])
ax[0].plot(x, G.mean(0), color=GREY, lw=2, ls="--", label="Grid (random visiting order)")
ax[0].axhline(DEF, color="k", lw=.8, ls=":"); ax[0].text(30, DEF - .0045, "library-default boosting (0.731)", fontsize=8)
ax[0].set(xlabel="evaluations used", ylabel="best CV ROC-AUC so far", title="Convergence (mean, band = min to max of 5 seeds)")
ax[0].legend(frameon=False, loc="center right", fontsize=8); ax[0].set_ylim(.725, .775)
data_, labs = [], []
for m in ["random", "optuna_tpe", "hyperopt_tpe"]:
    data_.append(T[T.method == m].groupby("seed").cv_auc.max().values); labs.append(NAMES[m].replace(" (", "\n("))
bp = ax[1].boxplot(data_, tick_labels=labs, patch_artist=True, widths=.5)
for b, c in zip(bp["boxes"], cols_.values()):
    b.set(facecolor=c, alpha=.55)
ax[1].axhline(gv.max(), color=GREY, ls="--", lw=1.2); ax[1].text(.55, gv.max() + .0003, "grid best", color=GREY, fontsize=8)
ax[1].set(ylabel="final best CV ROC-AUC", title="Final result per seed (81 evaluations)")
plt.tight_layout(); plt.savefig(OUT / "fig_convergence.png", dpi=150); plt.close()
print("saved fig_convergence.png")
