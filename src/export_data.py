"""Export the digits data and the exact split indices used in the paper to data/.

The experiment itself loads the digits data from scikit-learn and Fashion-MNIST from
data/fashion-mnist; these files are provided for transparency. Index conventions:
digits indices refer to rows of data/digits.csv; Fashion-MNIST indices 0-59999 refer to
the official training file and 60000-69999 to the official test file.
"""
import json
import numpy as np
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
header = ",".join([f"pixel_{i}" for i in range(64)] + ["label"])
np.savetxt("data/digits.csv", np.c_[X.astype(int), y], fmt="%d", delimiter=",", header=header, comments="")


for ds, seeds in (("digits", (0, 1, 2, 99)), ("fmnist", (0, 99))):
    for k in seeds:
        rng = np.random.RandomState(k)                       # identical to persistence_v2.py
        if ds == "digits":
            perm = rng.permutation(len(X))
            iD, iH, iP, iT = perm[:800], perm[800:1200], perm[1200:1600], perm[1600:]
            n_forget = 40
        else:
            perm = rng.permutation(60000)
            iD, iH, iP = perm[:5000], perm[5000:10000], perm[10000:15000]
            iT = 60000 + np.arange(10000)
            n_forget = 100
        fpos = rng.choice(len(iD), n_forget, replace=False)
        split = dict(dataset=ds, split_seed=k, note="split 99 is the hyperparameter-selection split",
                     train_D=iD.tolist(), forget_Df=iD[fpos].tolist(), retain_Dr=np.delete(iD, fpos).tolist(),
                     nonmember_H=iH.tolist(), public_pool_R=iP.tolist(), test=iT.tolist())
        json.dump(split, open(f"data/splits/{ds}_split{k}.json", "w"))
print("wrote data/digits.csv and data/splits/*.json")
