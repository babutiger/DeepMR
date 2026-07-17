#!/bin/bash
# ERAN psb batch 3 (reordered): prioritize 9x200 key data (DeepMR+DeepSRGR, reduced to 5 + timeout)
# 9x100 DeepSRGR already has 7, which is enough; skip the rest
source ~/anaconda3/etc/profile.d/conda.sh; conda activate py39deepsrgr
cd verify/mnist_new_6x100
run () {
  echo "===== $7 / $1 START ($(date '+%m-%d %H:%M')) ====="
  MODULE=$2 MODEL=../../models/$3 \
    PDIR=../../vnncomp_eran_properties/vnncomp_eran_mnist_properties_$4 PREF=mnist_property \
    EPS=$5 NPROPS=$6 TAG=$7 TIMEOUT=${8:-0} python3 -u $1 2>&1 | tail -2
  echo "===== $7 / $1 DONE ====="
}
# 9x200 DeepMR: 5 + 12000s timeout (200 network extremely slow; treat hang as DNF)
run eran_run_deepmr.py deepmr_mnist_new_6x100_3 mnist_new_9x200/mnist_9_200_nat_flat.nnet 9x200 0.012 5 eran9x200 12000
# 9x200 DeepSRGR: 5 + 8000s timeout (expected DNF)
run eran_run_srgr.py   deepsrgr_mnist_new_6x100 mnist_new_9x200/mnist_9_200_nat_flat.nnet 9x200 0.012 5 eran9x200 8000
echo "[ERAN_PSB3] ALL DONE"
