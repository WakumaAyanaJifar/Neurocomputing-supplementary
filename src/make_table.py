"""Aggregate the three splits into the rows of Table 10 (LaTeX), written to results/table_rows.tex."""
import json
import numpy as np

S = [json.load(open(f"results/results_split{k}.json"))["summary"] for k in range(3)]
by = [{e["component"]: e for e in s} for s in S]
agg = lambda name, key: np.array([b[name][key] for b in by])

rows = [("Original (before request)", "--", "T_orig", "T_cf"),
        ("Exactly unlearned (retrained)", "--", "T_unl", "T_cf")]
for c in "AB":
    rows += [("Stale (distilled before request)", c, f"{c}:S_stale", f"{c}:S_cf"),
             ("Stale, 8-bit weights", c, f"{c}:S_q8", f"{c}:S_cf"),
             ("Stale, 4-bit weights", c, f"{c}:S_q4", f"{c}:S_cf"),
             ("Re-distilled from unlearned teacher (R1)", c, f"{c}:S_new", f"{c}:S_cf")]
out = []
for desc, cond, name, ref in rows:
    adv, p = agg(name, "adv"), agg(name, "p")
    advs = f"{adv.min():.2f}" if adv.min() == adv.max() else f"{adv.min():.2f}--{adv.max():.2f}"
    ps = "$<10^{-10}$" if p.max() < 1e-10 else f"$\\geq {p.min():.2f}$"
    out.append(f"{desc} & {cond} & {agg(name, 'acc_mean').mean():.3f} & {agg(name, 'auc_mean').mean():.3f} & "
               f"{agg(ref, 'auc_mean').mean():.3f} & {agg(name, 'tpr5').mean():.2f} & {advs} & {ps} \\\\")
open("results/table_rows.tex", "w").write("\n".join(out) + "\n")
print("\n".join(out))
