#!/usr/bin/env bash
# Reproduce all results of Case G from the repository root.
set -e
python src/export_data.py
for k in 0 1 2; do
  python src/distill_persistence.py $k | tee results/results_split$k.txt
done
python src/make_table.py
python src/make_figure.py
