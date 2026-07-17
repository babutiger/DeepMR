#!/bin/bash
# ERAN psa batch 2: 6x200 DeepMR (reduced samples + timeout, avoid hangs) + 6x200 DeepSRGR (DNF)
source ~/anaconda3/etc/profile.d/conda.sh; conda activate py39deepsrgr
cd verify/mnist_new_6x100
run () {
  echo "===== $7 / $1 START ($(date '+%m-%d %H:%M')) ====="
  MODULE=$2 MODEL=../../models/$3 \
    PDIR=../../vnncomp_eran_properties/vnncomp_eran_mnist_properties_$4 PREF=mnist_property \
    EPS=$5 NPROPS=$6 TAG=$7 TIMEOUT=${8:-0} python3 -u $1 2>&1 | tail -2
  echo "===== $7 / $1 DONE ====="
}
# 6x200 DeepMR: 20 properties + 15000s timeout (avoid single-instance hangs)
run eran_run_deepmr.py deepmr_mnist_new_6x100_3 mnist_new_6x200/mnist_6_200_nat_flat.nnet 6x200 0.013 20 eran6x200 15000
# 6x200 DeepSRGR: 10 properties + 10000s timeout (expected DNF)
run eran_run_srgr.py   deepsrgr_mnist_new_6x100 mnist_new_6x200/mnist_6_200_nat_flat.nnet 6x200 0.013 10 eran6x200 10000
echo "[ERAN_PSA2] ALL DONE"
