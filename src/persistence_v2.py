"""
Case G, extended: persistence after exact and approximate unlearning, in teachers,
their quantized variants, and students distilled from them.

  python persistence_v2.py --dataset digits --split 0 --seeds 20
  python persistence_v2.py --dataset fmnist --split 0 --seeds 10
  python persistence_v2.py --dataset digits --select      # hyperparameter selection

Every component is compared with its OWN counterfactual: the same pipeline run on
D_r only (independent seeds), including the same quantization or distillation step.

Teachers:   orig (trained on D), exact (retrained on D_r), ga (gradient ascent /
            NegGrad+ from orig), scrub (SCRUB from orig); each also at 8 and 4 bits.
Students:   distilled from each teacher. Condition A: transfer data = the teacher's
            training data (D before the request, D_r after). Condition B: disjoint
            pool R. "stale" = distilled from orig before the request.
            (Fashion-MNIST runs condition B only, to fit a CPU budget.)
Tests:      T1 average loss-threshold AUC; T2 per-example calibrated TPR at 5% FPR;
            T3 model-level advantage (two-sample KS on the membership gap).
"""
import argparse, copy, gzip, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

torch.set_num_threads(2)
p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["digits", "fmnist"], required=True)
p.add_argument("--split", type=int, default=0)
p.add_argument("--seeds", type=int, default=20)
p.add_argument("--select", action="store_true", help="hyperparameter selection on split 99")
p.add_argument("--fmnist_dir", default="data/fashion-mnist")
p.add_argument("--out", default="results")
args = p.parse_args()
TEMP = 4.0

# ------------------------------------------------------------------ data
def load_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)

split_seed = 99 if args.select else args.split
rng = np.random.RandomState(split_seed)
if args.dataset == "digits":
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).astype(np.float32)
    perm = rng.permutation(len(X))
    iD, iH, iP, iT = perm[:800], perm[800:1200], perm[1200:1600], perm[1600:]
    n_forget = 40
    Xall, yall = X, y
else:
    d = args.fmnist_dir
    Xtr = load_idx(f"{d}/train-images-idx3-ubyte.gz"); ytr = load_idx(f"{d}/train-labels-idx1-ubyte.gz")
    Xte = load_idx(f"{d}/t10k-images-idx3-ubyte.gz"); yte = load_idx(f"{d}/t10k-labels-idx1-ubyte.gz")
    Xall = (np.concatenate([Xtr, Xte]).astype(np.float32) / 255.0)[:, None, :, :]
    yall = np.concatenate([ytr, yte]).astype(np.int64)
    perm = rng.permutation(60000)                  # D, H, R from the official training set
    iD, iH, iP = perm[:5000], perm[5000:10000], perm[10000:15000]
    iT = 60000 + np.arange(10000)                  # official test set
    n_forget = 100
fpos = rng.choice(len(iD), n_forget, replace=False)
iF, iR = iD[fpos], np.delete(iD, fpos)
T_ = lambda i: (torch.tensor(Xall[i]), torch.tensor(yall[i]).long())
XD, yD = T_(iD); XR, yR = T_(iR); XF, yF = T_(iF); XH, yH = T_(iH); XP, _ = T_(iP); XT, yT = T_(iT)

# ------------------------------------------------------------------ models and training
def teacher_net():
    if args.dataset == "digits":
        return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
    return nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3), nn.ReLU(),
                         nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 5 * 5, 256), nn.ReLU(), nn.Linear(256, 10))

def student_net():
    if args.dataset == "digits":
        return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    return nn.Sequential(nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3), nn.ReLU(),
                         nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(), nn.Linear(64, 10))

CFG = dict(digits=dict(t_epochs=300, s_epochs=400, bs=None, t_lr=1e-3, s_lr=2e-3),
           fmnist=dict(t_epochs=15, s_epochs=15, bs=128, t_lr=1e-3, s_lr=1e-3))[args.dataset]

def batches(n, bs, g):
    if bs is None:
        yield torch.arange(n); return
    idx = torch.randperm(n, generator=g)
    for k in range(0, n, bs):
        yield idx[k:k + bs]

def predict(m, X, bs=2000):
    with torch.no_grad():
        return torch.cat([m(X[k:k + bs]) for k in range(0, len(X), bs)])

def train_teacher(X, y, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = teacher_net(); opt = torch.optim.Adam(m.parameters(), lr=CFG["t_lr"])
    for _ in range(CFG["t_epochs"]):
        for b in batches(len(X), CFG["bs"], g):
            opt.zero_grad(); F.cross_entropy(m(X[b]), y[b]).backward(); opt.step()
    return m.eval()

def distill(teacher, X, seed):
    torch.manual_seed(seed + 50_000); g = torch.Generator().manual_seed(seed + 50_000)
    soft = F.softmax(predict(teacher, X) / TEMP, dim=1)
    s = student_net(); opt = torch.optim.Adam(s.parameters(), lr=CFG["s_lr"])
    for _ in range(CFG["s_epochs"]):
        for b in batches(len(X), CFG["bs"], g):
            opt.zero_grad()
            loss = F.kl_div(F.log_softmax(s(X[b]) / TEMP, 1), soft[b], reduction="batchmean") * TEMP ** 2
            loss.backward(); opt.step()
    return s.eval()

# ------------------------------------------------------------------ approximate unlearning
def ga_unlearn(m0, hp, seed):
    """Gradient ascent on D_f with descent on D_r (NegGrad+ style), Eq. (ga) of the paper."""
    torch.manual_seed(seed + 70_000); g = torch.Generator().manual_seed(seed + 70_000)
    m = copy.deepcopy(m0).train(); opt = torch.optim.SGD(m.parameters(), lr=hp["lr"], momentum=0.9)
    bs = CFG["bs"] or len(XR)
    for _ in range(hp["steps"]):
        bf = torch.randint(len(XF), (min(bs, len(XF)),), generator=g)
        br = torch.randint(len(XR), (bs,), generator=g)
        opt.zero_grad()
        loss = F.cross_entropy(m(XR[br]), yR[br]) - hp["alpha"] * F.cross_entropy(m(XF[bf]), yF[bf])
        loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    return m.eval()

def scrub_unlearn(m0, hp, seed):
    """SCRUB (Kurmanji et al., 2023): max-steps push the student away from the original
    model on D_f; min-steps keep it close on D_r (KL + cross-entropy)."""
    torch.manual_seed(seed + 80_000); g = torch.Generator().manual_seed(seed + 80_000)
    teacher = m0; s = copy.deepcopy(m0).train()
    opt = torch.optim.SGD(s.parameters(), lr=hp["lr"], momentum=0.9)
    bs = CFG["bs"] or len(XR)
    with torch.no_grad():
        pt_f = F.softmax(predict(teacher, XF), 1); pt_r = F.softmax(predict(teacher, XR), 1)
    for ep in range(hp["epochs"]):
        for _ in range(hp["rsteps"]):                        # min-steps on D_r
            br = torch.randint(len(XR), (bs,), generator=g)
            opt.zero_grad()
            out = s(XR[br])
            loss = F.kl_div(F.log_softmax(out, 1), pt_r[br], reduction="batchmean") + F.cross_entropy(out, yR[br])
            loss.backward(); opt.step()
        if ep < hp["msteps"]:                                # max-steps on D_f
            for b in batches(len(XF), CFG["bs"], g):
                opt.zero_grad()
                kl = F.kl_div(F.log_softmax(s(XF[b]), 1), pt_f[b], reduction="batchmean")
                (-hp["beta"] * kl).backward(); nn.utils.clip_grad_norm_(s.parameters(), 5.0); opt.step()
    return s.eval()

def quantize(model, bits):
    """Symmetric per-tensor quantization of all weight matrices and kernels (biases in float)."""
    q = copy.deepcopy(model); qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for mod in q.modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d)):
                sc = mod.weight.abs().max() / qmax
                mod.weight.copy_(torch.round(mod.weight / sc).clamp(-qmax, qmax) * sc)
    return q.eval()

# ------------------------------------------------------------------ measurement
def margin(logits, labels):
    zy = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.clone(); other[torch.arange(len(labels)), labels] = -float("inf")
    return (zy - torch.logsumexp(other, 1)).numpy()

def measure(m):
    oF, oH, oT = predict(m, XF).double(), predict(m, XH).double(), predict(m, XT)
    oF, oH = torch.nan_to_num(oF, 0.0, 1e4, -1e4), torch.nan_to_num(oH, 0.0, 1e4, -1e4)
    lf = F.cross_entropy(oF, yF, reduction="none").numpy(); lh = F.cross_entropy(oH, yH, reduction="none").numpy()
    auc = roc_auc_score(np.r_[np.ones(len(lf)), np.zeros(len(lh))], -np.r_[lf, lh])
    return dict(acc=(oT.argmax(1) == yT).float().mean().item(), facc=(oF.argmax(1) == yF).float().mean().item(),
                floss=float(lf.mean()), auc=float(auc), gap=float(lh.mean() - lf.mean()),
                mf=margin(oF, yF), mh=margin(oH, yH))

def per_example_tpr(actual, cf, fpr=0.05):
    cf_f = np.stack([r["mf"] for r in cf]); cf_h = np.stack([r["mh"] for r in cf])
    zf = np.concatenate([(r["mf"] - cf_f.mean(0)) / (cf_f.std(0) + 1e-6) for r in actual])
    zh = np.concatenate([(r["mh"] - cf_h.mean(0)) / (cf_h.std(0) + 1e-6) for r in actual])
    return float((zf > np.quantile(zh, 1 - fpr)).mean())

def adv(actual, cf):
    r = ks_2samp([x["gap"] for x in actual], [x["gap"] for x in cf]); return float(r.statistic), float(r.pvalue)

# ------------------------------------------------------------------ hyperparameter selection
GA_GRID = dict(digits=[dict(lr=lr, steps=st, alpha=a) for lr in (0.003, 0.01) for st in (10, 20, 30, 40, 60) for a in (0.5, 1.0)],
               fmnist=[dict(lr=lr, steps=st, alpha=a) for lr in (0.001, 0.003, 0.01) for st in (10, 30, 100) for a in (0.5, 1.0)])
SC_GRID = dict(digits=[dict(lr=lr, epochs=e, msteps=ms, rsteps=rs, beta=b) for lr in (0.01, 0.03) for (e, ms) in ((5, 3), (10, 6)) for rs in (5,) for b in (1.0, 5.0, 20.0)],
               fmnist=[dict(lr=lr, epochs=e, msteps=ms, rsteps=rs, beta=b) for lr in (0.01, 0.03) for (e, ms) in ((3, 2), (6, 4)) for rs in (20,) for b in (1.0, 5.0, 20.0)])

def select(n=3):
    """Choose the configuration whose mean forget-set loss is closest (in log scale) to that
    of retrained models, subject to test accuracy within 2 points of retrained models. This
    tuning uses the retrained reference and therefore favours the approximate methods."""
    origs = [train_teacher(XD, yD, s) for s in range(n)]
    cfs = [measure(train_teacher(XR, yR, s + 2000)) for s in range(n)]
    target_l, target_a = np.mean([c["floss"] for c in cfs]), np.mean([c["acc"] for c in cfs])
    out = {}
    for name, fn, grid in (("ga", ga_unlearn, GA_GRID), ("scrub", scrub_unlearn, SC_GRID)):
        scored = []
        for hp in grid[args.dataset]:
            ms = [measure(fn(o, hp, s)) for s, o in enumerate(origs)]
            l, a = np.mean([m["floss"] for m in ms]), np.mean([m["acc"] for m in ms])
            ok = bool(np.isfinite(l)) and a >= target_a - 0.02
            scored.append((not ok, abs(np.log(l + 1e-9) - np.log(target_l)), hp, l, a))
            print(name, hp, f"floss={l:.4f} acc={a:.4f} ok={ok}", file=sys.stderr, flush=True)
        scored.sort(key=lambda t: (t[0], t[1]))
        out[name] = dict(hp=scored[0][2], floss=scored[0][3], acc=scored[0][4])
    out["target"] = dict(floss=target_l, acc=target_a)
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, open(f"{args.out}/{args.dataset}_selected_hparams.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if args.select:
    select(); sys.exit()

HP = json.load(open(f"{args.out}/{args.dataset}_selected_hparams.json"))

# ------------------------------------------------------------------ main experiment
rows = {}
def add(name, m): rows.setdefault(name, []).append(measure(m))
t0 = time.time()
for s in range(args.seeds):
    T = dict(orig=train_teacher(XD, yD, s), exact=train_teacher(XR, yR, s + 1000))
    T["ga"] = ga_unlearn(T["orig"], HP["ga"]["hp"], s)
    T["scrub"] = scrub_unlearn(T["orig"], HP["scrub"]["hp"], s)
    Tcf = train_teacher(XR, yR, s + 2000)
    for k, m in T.items():
        add(f"T:{k}", m); add(f"T:{k}:q8", quantize(m, 8)); add(f"T:{k}:q4", quantize(m, 4))
    add("T:cf", Tcf); add("T:cf:q8", quantize(Tcf, 8)); add("T:cf:q4", quantize(Tcf, 4))
    conds = {"A": (XD, XR), "B": (XP, XP)} if args.dataset == "digits" else {"B": (XP, XP)}
    for cond, (Xb, Xa) in conds.items():
        stale = distill(T["orig"], Xb, s)
        add(f"{cond}:S:stale", stale); add(f"{cond}:S:stale:q4", quantize(stale, 4))
        for k in ("exact", "ga", "scrub"):
            add(f"{cond}:S:{k}", distill(T[k], Xa, s + 1000))
        cf = distill(Tcf, Xa, s + 2000)
        add(f"{cond}:S:cf", cf); add(f"{cond}:S:cf:q4", quantize(cf, 4))
    print(f"seed {s + 1}/{args.seeds} done ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

def ref_of(name):
    parts = name.split(":")
    if parts[0] == "T":
        return "T:cf" + (":" + parts[2] if len(parts) > 2 else "")
    return f"{parts[0]}:S:cf" + (":" + parts[3] if len(parts) > 3 else "")

summary = []
for name, r in rows.items():
    e = {k: float(np.mean([x[k] for x in r])) for k in ("acc", "facc", "floss", "auc", "gap")}
    e.update(component=name, auc_sd=float(np.std([x["auc"] for x in r])))
    if ":cf" not in name:
        ref = ref_of(name)
        e["ref"] = ref; e["auc_ref"] = float(np.mean([x["auc"] for x in rows[ref]]))
        e["adv"], e["p"] = adv(r, rows[ref]); e["tpr5"] = per_example_tpr(r, rows[ref])
    summary.append(e)
os.makedirs(args.out, exist_ok=True)
tag = f"{args.dataset}_split{args.split}"
json.dump(dict(config=dict(dataset=args.dataset, split=args.split, seeds=args.seeds, n_forget=n_forget,
                           sizes=dict(D=len(iD), H=len(iH), R=len(iP), test=len(iT)), hparams=HP, cfg=CFG),
               summary=summary), open(f"{args.out}/{tag}.json", "w"), indent=2)
np.savez(f"{args.out}/{tag}_gaps.npz", **{k: np.array([x["gap"] for x in v]) for k, v in rows.items()})
print(f"{'component':16s} {'acc':>6s} {'f-acc':>6s} {'AUC':>6s} {'AUCcf':>6s} {'TPR5':>5s} {'Adv':>5s} {'p':>8s}")
for e in summary:
    print(f"{e['component']:16s} {e['acc']:.3f} {e['facc']:.3f} {e['auc']:.3f} "
          + (f"{e['auc_ref']:.3f} {e['tpr5']:.2f} {e['adv']:.2f} {e['p']:.1e}" if "adv" in e else ""))
