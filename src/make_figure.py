"""Figure for Case G: membership-gap statistic per training seed (split 0)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})
g = np.load("results/gaps_split0.npz")
items = [("Teacher, original", "T_orig", "T_cf", True),
         ("Teacher, exactly unlearned", "T_unl", "T_cf", False),
         ("A: student, stale", "A:S_stale", "A:S_cf", True),
         ("A: student, stale, 4-bit", "A:S_q4", "A:S_cf", True),
         ("A: student, re-distilled", "A:S_new", "A:S_cf", False),
         ("B: student, stale", "B:S_stale", "B:S_cf", True),
         ("B: student, stale, 4-bit", "B:S_q4", "B:S_cf", True),
         ("B: student, re-distilled", "B:S_new", "B:S_cf", False)]
RED, BLUE, GRAY = "#C62828", "#1F5AA6", "#9CA3AF"
fig, ax = plt.subplots(figsize=(5.6, 3.1))
rng = np.random.RandomState(0)
for i, (lab, k, ref, persists) in enumerate(items):
    y = len(items) - 1 - i
    ax.scatter(g[ref], y - 0.17 + rng.uniform(-0.06, 0.06, len(g[ref])), s=9, color=GRAY, lw=0, zorder=2)
    ax.scatter(g[k], y + 0.17 + rng.uniform(-0.06, 0.06, len(g[k])), s=9,
               color=RED if persists else BLUE, lw=0, zorder=3)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([it[0] for it in items][::-1])
ax.axvline(0, color="#D1D5DB", lw=0.6, zorder=1)
for yy in (2.5, 5.5):
    ax.axhline(yy, color="#E5E7EB", lw=0.6)
ax.set_xlabel("Membership gap = mean non-member loss $-$ mean loss on $D_f$ (dots: seeds)")
ax.spines[["top", "right"]].set_visible(False)
h = [plt.Line2D([], [], marker="o", ls="", color=c, ms=3.5, label=l) for c, l in
     [(RED, "component, persistence detected"), (BLUE, "component, not detected"), (GRAY, "counterfactual")]]
ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, fontsize=6.5, handletextpad=0.2, columnspacing=1.0)
fig.tight_layout()
fig.savefig("results/fig_experiment.pdf", bbox_inches="tight")
fig.savefig("results/fig_experiment.png", dpi=200, bbox_inches="tight")
