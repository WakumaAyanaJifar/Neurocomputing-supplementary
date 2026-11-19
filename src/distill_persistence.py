"""
Illustrative experiment for the tutorial: does exact unlearning of a teacher
remove persistence from students distilled before the request?

Operationalizes Definition 1 (persistence as distinguishability from the
counterfactual state) and Proposition 1 (certified unlearning does not transfer
through distillation).

Data: scikit-learn's bundled copy of the UCI optical handwritten digits data
(1,797 8x8 images, 10 classes); no download is needed.

Components evaluated (for each training seed):
  T_orig   teacher trained on D (before the request)
  T_unl    teacher after exact unlearning (retrained on D_r; eps = delta = 0)
  S_stale  student distilled from T_orig before the request, never updated
  S_q8/q4  S_stale after 8-bit / 4-bit weight quantization (deployed variants)
  S_new    student re-distilled from T_unl (remedy R1)
Two distillation conditions:
  A: transfer set = the teacher's training data (D before, D_r after)
  B: transfer set = a disjoint public pool P (independent of D_f, as in Prop. 1)
Counterfactual references are drawn with independent seeds from the pipeline
run on D_r only.

Run from the repository root:  python src/distill_persistence.py <split>
(split = 0, 1, or 2; CPU, about 1.5 minutes per split). Outputs go to results/.
"""
import json, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

torch.set_num_threads(2)
SPLIT_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_SEEDS = 20
N_FORGET = 40
TEMP = 4.0

# ------------------------------------------------------------------ data
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
rng = np.random.RandomState(SPLIT_SEED)
perm = rng.permutation(len(X))
idx_D, idx_H, idx_P, idx_test = perm[:800], perm[800:1200], perm[1200:1600], perm[1600:]
forget_pos = rng.choice(len(idx_D), N_FORGET, replace=False)
idx_F = idx_D[forget_pos]
idx_R = np.delete(idx_D, forget_pos)

t = lambda i: (torch.tensor(X[i]), torch.tensor(y[i]))
XD, yD = t(idx_D); XR, yR = t(idx_R); XF, yF = t(idx_F)
XH, yH = t(idx_H); XP, _ = t(idx_P); XT, yT = t(idx_test)


def mlp(width, depth):
    layers, d = [], 64
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, 10))
    return nn.Sequential(*layers)


def train_teacher(Xtr, ytr, seed, epochs=300):
    torch.manual_seed(seed)
    m = mlp(256, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(m(Xtr), ytr).backward()
        opt.step()
    return m.eval()


def distill(teacher, Xtr, seed, epochs=400):
    torch.manual_seed(seed + 50_000)
    with torch.no_grad():
        soft = F.softmax(teacher(Xtr) / TEMP, dim=1)
    s = mlp(64, 1)
    opt = torch.optim.Adam(s.parameters(), lr=2e-3)
    for _ in range(epochs):
        opt.zero_grad()
        logp = F.log_softmax(s(Xtr) / TEMP, dim=1)
        (F.kl_div(logp, soft, reduction="batchmean") * TEMP ** 2).backward()
        opt.step()
    return s.eval()


def quantize(model, bits):
    """Symmetric per-tensor weight quantization (biases kept in float)."""
    q = mlp(64, 1)
    q.load_state_dict(model.state_dict())
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for mod in q:
            if isinstance(mod, nn.Linear):
                scale = mod.weight.abs().max() / qmax
                mod.weight.copy_(torch.round(mod.weight / scale).clamp(-qmax, qmax) * scale)
    return q.eval()


def margin(logits, labels):
    """Stable LiRA score log(p_y/(1-p_y)) = z_y - logsumexp(z_{k != y})."""
    zy = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.clone(); other[torch.arange(len(labels)), labels] = -float("inf")
    return (zy - torch.logsumexp(other, 1)).numpy()


def measure(model):
    with torch.no_grad():
        lf = F.cross_entropy(model(XF), yF, reduction="none").numpy()
        lh = F.cross_entropy(model(XH), yH, reduction="none").numpy()
        acc = (model(XT).argmax(1) == yT).float().mean().item()
    labels = np.r_[np.ones(len(lf)), np.zeros(len(lh))]
    auc = roc_auc_score(labels, -np.r_[lf, lh])  # loss-threshold MIA
    gap = float(lh.mean() - lf.mean())          # membership gap statistic
    with torch.no_grad():
        mf, mh = margin(model(XF).double(), yF), margin(model(XH).double(), yH)
    return dict(acc=acc, auc=float(auc), gap=gap, mf=mf, mh=mh)


rows = {}
def add(name, m):
    rows.setdefault(name, []).append(measure(m))

for s in range(N_SEEDS):
    # actual pipeline (data entered at training time), then request handled
    T_orig = train_teacher(XD, yD, seed=s)
    T_unl = train_teacher(XR, yR, seed=s + 1000)           # exact unlearning
    # counterfactual pipeline (D_f never entered), independent seeds
    T_cf = train_teacher(XR, yR, seed=s + 2000)
    add("T_orig", T_orig); add("T_unl", T_unl); add("T_cf", T_cf)
    for cond, (X_before, X_after) in {"A": (XD, XR), "B": (XP, XP)}.items():
        S_stale = distill(T_orig, X_before, seed=s)
        S_new = distill(T_unl, X_after, seed=s + 1000)
        S_cf = distill(T_cf, X_after, seed=s + 2000)
        add(f"{cond}:S_stale", S_stale)
        add(f"{cond}:S_q8", quantize(S_stale, 8))
        add(f"{cond}:S_q4", quantize(S_stale, 4))
        add(f"{cond}:S_new", S_new)
        add(f"{cond}:S_cf", S_cf)
    print(f"seed {s+1}/{N_SEEDS} done", file=sys.stderr, flush=True)


def logit_conf(loss):
    """LiRA-style scaling: log(p/(1-p)) of the true-class probability."""
    p = np.clip(np.exp(-loss), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def per_example_tpr(actual, cf, fpr=0.05):
    """Offline per-example likelihood-ratio test in the spirit of LiRA
    (Carlini et al., 2022): each example's score under the component is
    standardized against its score distribution under counterfactual models.
    The threshold is set on non-members H to the given false-positive rate."""
    cf_f = np.stack([r["mf"] for r in cf]); cf_h = np.stack([r["mh"] for r in cf])
    mu_f, sd_f = cf_f.mean(0), cf_f.std(0) + 1e-6
    mu_h, sd_h = cf_h.mean(0), cf_h.std(0) + 1e-6
    zf = np.concatenate([(r["mf"] - mu_f) / sd_f for r in actual])
    zh = np.concatenate([(r["mh"] - mu_h) / sd_h for r in actual])
    thr = np.quantile(zh, 1 - fpr)
    return float((zf > thr).mean())


def adv(actual, cf):
    """Empirical Adv over the threshold family on the gap statistic (= KS)."""
    a = [r["gap"] for r in actual]; c = [r["gap"] for r in cf]
    res = ks_2samp(a, c)
    return float(res.statistic), float(res.pvalue)

summary = []
order = [("T_orig", "T_cf"), ("T_unl", "T_cf")]
for cond in "AB":
    order += [(f"{cond}:{k}", f"{cond}:S_cf") for k in ("S_stale", "S_q8", "S_q4", "S_new")]
order += [("T_cf", None)] + [(f"{c}:S_cf", None) for c in "AB"]
for name, ref in order:
    r = rows[name]
    entry = dict(component=name,
                 acc_mean=np.mean([x["acc"] for x in r]), acc_sd=np.std([x["acc"] for x in r]),
                 auc_mean=np.mean([x["auc"] for x in r]), auc_sd=np.std([x["auc"] for x in r]),
                 gap_mean=np.mean([x["gap"] for x in r]), gap_sd=np.std([x["gap"] for x in r]))
    if ref:
        entry["adv"], entry["p"] = adv(r, rows[ref])
        entry["tpr5"] = per_example_tpr(r, rows[ref])
    summary.append(entry)

print(f"{'component':12s} {'acc':>13s} {'MIA AUC':>13s} {'gap':>15s} {'Adv':>6s} {'p':>9s} {'TPR@5%':>7s}")
for e in summary:
    a = f"{e['adv']:.2f}" if "adv" in e else "-"
    p = f"{e['p']:.1e}" if "p" in e else "-"
    tp = f"{e['tpr5']:.2f}" if "tpr5" in e else "-"
    print(f"{e['component']:12s} {e['acc_mean']:.3f}±{e['acc_sd']:.3f} "
          f"{e['auc_mean']:.3f}±{e['auc_sd']:.3f} {e['gap_mean']:.4f}±{e['gap_sd']:.4f} {a:>6s} {p:>9s} {tp:>7s}")
json.dump(dict(config=dict(split_seed=SPLIT_SEED, n_seeds=N_SEEDS, n_forget=N_FORGET,
                           temperature=TEMP, n_D=800, n_H=400, n_P=400, n_test=len(idx_test)),
               summary=summary), open(f"results/results_split{SPLIT_SEED}.json", "w"), indent=2, default=float)
np.savez(f"results/gaps_split{SPLIT_SEED}.npz", **{k: np.array([x["gap"] for x in v]) for k, v in rows.items()})
