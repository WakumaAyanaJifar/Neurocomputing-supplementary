"""Export the digits data and the exact split indices used in the paper to data/.

The experiment itself loads the data from scikit-learn; these files are provided
for transparency and for use outside Python.
"""
import json
import numpy as np
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
header = ",".join([f"pixel_{i}" for i in range(64)] + ["label"])
np.savetxt("data/digits.csv", np.c_[X.astype(int), y], fmt="%d", delimiter=",", header=header, comments="")
for k in range(3):
    rng = np.random.RandomState(k)                      # identical to distill_persistence.py
    perm = rng.permutation(len(X))
    idx_D, idx_H, idx_P, idx_test = perm[:800], perm[800:1200], perm[1200:1600], perm[1600:]
    forget_pos = rng.choice(len(idx_D), 40, replace=False)
    split = dict(split_seed=k,
                 train_D=idx_D.tolist(), forget_Df=idx_D[forget_pos].tolist(),
                 retain_Dr=np.delete(idx_D, forget_pos).tolist(),
                 nonmember_H=idx_H.tolist(), public_pool_R=idx_P.tolist(), test=idx_test.tolist())
    json.dump(split, open(f"data/splits/split{k}.json", "w"))
print("wrote data/digits.csv and data/splits/split{0,1,2}.json")
