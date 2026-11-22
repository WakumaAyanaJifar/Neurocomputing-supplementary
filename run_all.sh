#!/usr/bin/env bash
# Reproduce all results of Case G from the repository root (CPU; about 1.5 hours in total).
set -e
mkdir -p results
python src/export_data.py
python src/persistence_v2.py --dataset digits --select > results/digits_select.txt 2> results/digits_select_grid.txt
for k in 0 1 2; do
  python src/persistence_v2.py --dataset digits --split $k --seeds 20 > results/digits_split$k.txt
done
python src/persistence_v2.py --dataset fmnist --select > results/fmnist_select.txt 2> results/fmnist_select_grid.txt
python src/persistence_v2.py --dataset fmnist --split 0 --seeds 10 > results/fmnist_split0.txt
python src/make_tables_v2.py digits fmnist
python src/make_figure_v2.py
