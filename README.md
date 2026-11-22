# Persistence of learned information after unlearning: teachers, quantized variants, and students

Code and data for **Case G** of the tutorial

> J. W. Ayana and H. Ning, "Beyond Single-Model Unlearning: A Tutorial on Learned-Information
> Persistence Across Cyber-Physical-Social-Thinking Spaces," submitted to *Neurocomputing*, 2026.

The experiment evaluates every component against **its own counterfactual**: the same pipeline
run without the forget set, including the same quantization or distillation step. It asks three
questions:

1. Does exact unlearning of a teacher remove the persistence carried by students distilled from
   it before the request (Proposition 1)?
2. Do approximate methods (gradient ascent and SCRUB) leave the teacher distinguishable from
   retraining, and does 4-bit quantization restore what they removed?
3. Does re-distilling a student from an unlearned teacher remove persistence?

## Main results

**Digits** (3 splits × 20 seeds) and **Fashion-MNIST** (1 split × 10 seeds):

- **Stale students.** Students distilled before the request stay distinguishable from their
  counterfactuals after exact unlearning of the teacher, and so do their 4-bit variants. This
  holds even when the student was distilled on data disjoint from the forget set.
- **Gradient ascent.** Tuned to keep test accuracy within 2 points, it leaves the teacher
  distinguishable on both data sets.
- **SCRUB.** It matches the counterfactual forget-set accuracy, yet stays distinguishable on 2 of 3
  digits splits. Across seeds it both over-forgets and under-forgets.
- **Quantization.** 4-bit quantization neither restored nor removed persistence in these small
  models.
- **Re-distillation.** It passes on what its source still carries. Students re-distilled from an
  exactly unlearned teacher are never detected. Students re-distilled from gradient-ascent
  teachers are detected on digits but not on Fashion-MNIST.
- **Average attacks.** The average loss-threshold attack gives AUC 0.51–0.54 for components that
  clearly persist, and the counterfactual AUC is itself about 0.51–0.52.

## Repository layout

```
src/
  persistence_v2.py   main experiment (--select for hyperparameter selection; --split, --seeds)
  make_tables_v2.py   LaTeX rows of the two result tables
  make_figure_v2.py   the figure (membership gap per seed, both data sets)
  export_data.py      writes digits.csv and the exact split indices
data/
  digits.csv               the 1,797 digit images (64 pixel values, 0-16) and labels
  fashion-mnist/           the four official Fashion-MNIST files (unmodified)
  splits/*.json            indices of D, D_f, D_r, H, R and the test set for every split
results/                   outputs of the reported runs: per-split JSON and text, selected
                           hyperparameters, selection grids, per-seed gaps (NPZ), tables, figure
run_all.sh                 reproduces everything
```

## Reproduce

```bash
pip install -r requirements.txt
./run_all.sh
```

The scripts run on CPU. Each digits split takes about 3 minutes and Fashion-MNIST takes about
20 minutes, plus the two hyperparameter-selection runs. The reported runs used Python 3.13,
torch 2.15, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.3 and matplotlib 3.11.2. With these
versions, rerunning reproduces `results/` exactly. Other versions may differ in the last digits.

## Design

| Item | Digits | Fashion-MNIST |
|---|---|---|
| Train D / non-members H / pool R / test | 800 / 400 / 400 / 197 | 5,000 / 5,000 / 5,000 / 10,000 (official test set) |
| Forget set D_f | 40 random images of D | 100 random images of D |
| Teacher | MLP 64-256-256-10, 300 full-batch epochs | CNN (conv 32, conv 64, FC 256), 15 epochs |
| Student | MLP 64-64-10 | CNN (conv 16, conv 32, FC 64) |
| Distillation | temperature 4; conditions A and B | temperature 4; condition B only |
| Seeds × splits | 20 × 3 | 10 × 1 |

In distillation condition A the transfer data are the teacher's training data. In condition B
they are the disjoint pool R. Teachers use Adam with no regularization.

**Unlearning.** Three methods are compared:

- **Exact:** retraining on D_r.
- **Gradient ascent:** ascent on D_f with descent on D_r, using SGD with momentum and gradient
  clipping.
- **SCRUB:** alternating min-steps on D_r (KL to the original model plus cross-entropy) and
  max-steps on D_f.

Hyperparameters were selected on a separate split (seed 99). The rule picks the grid point whose
forget-set loss is closest to that of retrained models, subject to test accuracy within 2 points
of retrained models. See `results/*_selected_hparams.json` and `results/*_select_grid.txt`.

**Quantization.** Symmetric per-tensor quantization of all weight matrices and kernels, at 8 and
4 bits. Each quantized component is compared with counterfactuals quantized the same way.

**Tests.** Three test families are applied to every component:

- **T1, average attack:** the AUC of a loss-threshold attack separating D_f from H.
- **T2, per-example test:** each example's logit-scaled confidence is standardized against the
  counterfactual models (LiRA-style). We report the true-positive rate at 5% false-positive
  rate; 0.05 is the value expected by chance.
- **T3, model-level test:** a two-sample Kolmogorov–Smirnov test on the membership gap (mean
  non-member loss minus mean forget-set loss) across seeds. The KS statistic estimates Adv in
  Definition 1, and detection means p < 0.05.

## Limitations

The models and data sets are small, and Fashion-MNIST was run on one split with condition B only.
The approximate methods were tuned with one criterion. We did not run ResNet-18 on CIFAR-10: it
needs many counterfactual models per arm, which was beyond our computing budget. To try it, swap
the data loader and the `teacher_net`/`student_net` definitions in `src/persistence_v2.py`.

## Data and licenses

The code is released under the MIT License (see `LICENSE`).

- **Optical Recognition of Handwritten Digits:** E. Alpaydin and C. Kaynak, UCI Machine
  Learning Repository, 1998, https://doi.org/10.24432/C50P49, licensed CC BY 4.0.
  `data/digits.csv` is an unmodified export of the copy bundled with scikit-learn.
- **Fashion-MNIST:** H. Xiao, K. Rasul and R. Vollgraf, arXiv:1708.07747, 2017, MIT License,
  © 2017 Zalando SE. The files in `data/fashion-mnist/` are the unmodified official files from
  https://github.com/zalandoresearch/fashion-mnist. Their MD5 checksums match the official
  README: train images 8d4fb7e6c68d591d4c3dfef9ec88bf0d, train labels
  25c81989df183df01b3e8a0aad5dffbe, test images bef4ecab320f06d8554ea6380940ec79, test labels
  bb300cfdad3c16e7a12a480ee83cd310.

## Citation

If you use this code, please cite the paper above. The citation will be updated with publication
details.
