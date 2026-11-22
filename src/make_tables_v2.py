"""Build the LaTeX rows of the Case G tables from results/<dataset>_split<k>.json."""
import json, sys, glob
import numpy as np

LABELS = {"orig": "Original (before request)", "exact": "Exact unlearning (retrained)",
          "ga": "Gradient ascent", "scrub": "SCRUB"}
SLAB = {"stale": "Stale (distilled before request)", "exact": "Re-distilled from exact",
        "ga": "Re-distilled from gradient ascent", "scrub": "Re-distilled from SCRUB"}

def rows_for(dataset):
    files = sorted(glob.glob(f"results/{dataset}_split*.json"))
    S = [{e["component"]: e for e in json.load(open(f))["summary"]} for f in files]
    def line(label, cond, name):
        g = lambda k: np.array([s[name][k] for s in S])
        ref = S[0][name]["ref"]
        cf_facc = np.mean([s[ref]["facc"] for s in S])
        adv, p = g("adv"), g("p")
        advs = f"{adv.min():.2f}" if np.isclose(adv.min(), adv.max()) else f"{adv.min():.2f}--{adv.max():.2f}"
        ps = f"{int((p < 0.05).sum())}/{len(p)}"
        return (f"{label} & {cond} & {g('acc').mean():.3f} & {g('facc').mean():.3f} ({cf_facc:.3f}) & "
                f"{g('auc').mean():.3f} ({g('auc_ref').mean():.3f}) & {g('tpr5').mean():.2f} & {advs} & {ps} \\\\")
    out = ["\\multicolumn{8}{l}{\\emph{Teachers}} \\\\"]
    for k in ("orig", "exact", "ga", "scrub"):
        out.append(line(LABELS[k], "--", f"T:{k}"))
        out.append(line(LABELS[k] + ", 4-bit", "--", f"T:{k}:q4"))
    conds = [c for c in "AB" if f"{c}:S:stale" in S[0]]
    for c in conds:
        out.append("\\midrule")
        out.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{Students, condition {c}}}}} \\\\")
        out.append(line(SLAB["stale"], c, f"{c}:S:stale"))
        out.append(line(SLAB["stale"] + ", 4-bit", c, f"{c}:S:stale:q4"))
        for k in ("exact", "ga", "scrub"):
            out.append(line(SLAB[k], c, f"{c}:S:{k}"))
    return "\n".join(out), len(files)

for ds in sys.argv[1:]:
    txt, n = rows_for(ds)
    open(f"results/table_{ds}.tex", "w").write(txt + "\n")
    print(f"% {ds}: {n} split(s)\n{txt}\n")
