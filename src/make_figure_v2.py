"""Fig. 3: membership gap per seed for each component and its own counterfactual
(digits, split 0, and Fashion-MNIST, split 0)."""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "axes.linewidth": 0.6, "pdf.fonttype": 42})
RED, BLUE, GRAY = "#C62828", "#1F5AA6", "#9CA3AF"
ITEMS = [("Teacher, original", "T:orig"), ("Teacher, exact unlearning", "T:exact"),
         ("Teacher, gradient ascent", "T:ga"), ("Teacher, gradient ascent, 4-bit", "T:ga:q4"),
         ("Teacher, SCRUB", "T:scrub"), ("Teacher, SCRUB, 4-bit", "T:scrub:q4"),
         ("B: student, stale", "B:S:stale"), ("B: student from exact", "B:S:exact"),
         ("B: student from grad. ascent", "B:S:ga"), ("B: student from SCRUB", "B:S:scrub")]

def panel(ax, dataset, title):
    g = np.load(f"results/{dataset}_split0_gaps.npz")
    summ = {e["component"]: e for e in json.load(open(f"results/{dataset}_split0.json"))["summary"]}
    rng = np.random.RandomState(0)
    for i, (lab, k) in enumerate(ITEMS):
        y = len(ITEMS) - 1 - i
        ref = summ[k]["ref"]
        col = RED if summ[k]["p"] < 0.05 else BLUE
        ax.scatter(g[ref], y - 0.17 + rng.uniform(-0.05, 0.05, len(g[ref])), s=7, color=GRAY, lw=0)
        ax.scatter(g[k], y + 0.17 + rng.uniform(-0.05, 0.05, len(g[k])), s=7, color=col, lw=0)
    ax.set_yticks(range(len(ITEMS))); ax.set_yticklabels([l for l, _ in ITEMS][::-1])
    for yy in (3.5, 5.5, 7.5):
        ax.axhline(yy, color="#E5E7EB", lw=0.6)
    ax.set_title(title, fontsize=7.6)
    ax.set_xlabel("Membership gap (dots: seeds)")
    ax.spines[["top", "right"]].set_visible(False)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
panel(axes[0], "digits", "Digits (MLP), split 0, 20 seeds")
panel(axes[1], "fmnist", "Fashion-MNIST (CNN), 10 seeds")
h = [plt.Line2D([], [], marker="o", ls="", color=c, ms=3.5, label=l) for c, l in
     [(RED, "component, distinguishable (p < 0.05, T3)"), (BLUE, "component, not distinguishable"), (GRAY, "its counterfactual")]]
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, fontsize=6.8, bbox_to_anchor=(0.55, 1.08))
fig.tight_layout()
fig.savefig("results/fig_experiment.pdf", bbox_inches="tight")
fig.savefig("results/fig_experiment.png", dpi=200, bbox_inches="tight")
