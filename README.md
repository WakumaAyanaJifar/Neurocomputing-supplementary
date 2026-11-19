# Persistence of learned information after unlearning: distilled students

Code and data for **Case G** of the tutorial

> J. W. Ayana ,J.Ding and H. Ning, "Beyond Single-Model Unlearning: A Tutorial on Learned-Information
> Persistence in Neural Learning Systems Across Cyber-Physical-Social-Thinking Spaces,"
> submitted to *Neurocomputing*, 2026.

The experiment asks one question: **does exact unlearning of a teacher network remove the
information that students distilled from it before the deletion request still carry?**
It demonstrates Proposition 1 of the paper, which says that certified unlearning does not transfer
through distillation. It also shows how Definition 1, persistence as distinguishability from a
counterfactual state, is used in practice.

## Main result

After the teacher is unlearned exactly, by retraining without the forget set, the outcome depends
on the student:

- **Stale students** (distilled before the request) stay fully distinguishable from their
  counterfactuals, with model-level advantage 1.00 and p < 1e-10, on all three splits.
- **Quantized stale students** (8-bit and 4-bit) are just as distinguishable.
- **Disjoint transfer data** does not help: the result holds even when the student was distilled on
  data that never contained the forget set.
- **Re-distilled students** cannot be told apart from their counterfactuals by any of the three
  tests.

An average loss-threshold membership attack gives AUC 0.51–0.54 for the components that still
carry the information. A naive evaluation would read that as forgetting. Tests that compare each
component with its own counterfactual reveal the persistence.

## Repository layout

```
src/
  distill_persistence.py   main experiment (one data split per run)
  make_table.py            aggregates the three splits into Table 10 (LaTeX rows)
  make_figure.py           Fig. 3 (membership gap per seed, split 0)
  export_data.py           writes the data and exact split indices to data/
data/
  digits.csv               the 1,797 digit images (64 pixel values, 0-16) and labels
  splits/split{0,1,2}.json indices of D, D_f, D_r, H, R, and the test set for each split
results/                   outputs of the reported runs (JSON, text, NPZ, table, figure)
run_all.sh                 reproduces everything
```

## Reproduce

```bash
pip install -r requirements.txt
./run_all.sh
```

The script runs on CPU and takes about 5 minutes in total. The reported runs used Python 3, torch
2.15, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.3 and matplotlib 3.11.2. With these versions,
rerunning reproduces `results/` exactly. Other versions may differ in the last digits.

## Design in brief

| Item | Setting |
|---|---|
| Data | UCI optical handwritten digits, loaded from scikit-learn (no download) |
| Split | 800 training (D), 400 non-members (H), 400 public pool (R), 197 test; forget set D_f = 40 random images of D |
| Teacher | MLP 64-256-256-10, Adam, 300 full-batch epochs, no regularization |
| Student | MLP 64-64-10, distilled at temperature 4 for 400 epochs |
| Condition A | Transfer data = the teacher's training data |
| Condition B | Transfer data = the disjoint public pool R (independent of D_f) |
| Unlearning | Exact: the teacher is retrained on D_r |
| Counterfactuals | The same pipeline run on D_r only, with independent seeds |
| Seeds | 20 per component and per counterfactual; 3 random splits |

The script compares six components with their counterfactuals:

- the original teacher;
- the unlearned teacher;
- the stale student;
- the stale student with weights quantized to 8 bits;
- the stale student with weights quantized to 4 bits;
- the student re-distilled from the unlearned teacher.

It applies three test families:

- **T1, average membership attack:** the AUC of a loss-threshold attack separating D_f from H.
- **T2, per-example test:** each example's logit-scaled confidence is standardized against its
  distribution under the counterfactual models (in the spirit of LiRA). We report the
  true-positive rate at 5% false-positive rate; 0.05 is the value expected by chance.
- **T3, model-level test:** threshold tests on the membership gap (mean non-member loss minus mean
  forget-set loss), computed across seeds. This estimates the advantage Adv in Definition 1 and
  equals the two-sample Kolmogorov–Smirnov statistic.

## Limitations

The data and models are small, and the teacher is unlearned exactly, not approximately. Tests T2
and T3 need counterfactual models, which only someone with access to D_r can train. The
experiment illustrates a procedure; it is not a benchmark.

## Data and license

The code is released under the MIT License (see `LICENSE`).

The digits data are by E. Alpaydin and C. Kaynak, "Optical Recognition of Handwritten Digits",
UCI Machine Learning Repository, 1998, https://doi.org/10.24432/C50P49. They are licensed under
CC BY 4.0. `data/digits.csv` is an unmodified export of the copy bundled with scikit-learn.

## Citation

If you use this code, please cite the paper above. The citation will be updated with publication
details.
