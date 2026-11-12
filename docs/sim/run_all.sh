#!/usr/bin/env bash
# Regenerates docs/sim/*.csv (about 20 minutes on 6 cores). Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/../../scripts"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=../docs/sim
M=plugin,auto_rich,explicit,auto5,explicit5,lin_plugin,lin_auto3,lin_explicit,lin_auto3_5
for D in none tinv tvar; do
  python sim_readingb.py --design "$D" --reps 200 --n-units 1000 --T 5 --rho 0.3 0.9 --methods "$M" \
    --workers 6 --truth-units 300000 --out "$OUT/mc_$D.csv"
done
python sim_readingb.py --design tvar --reps 200 --n-units 1000 --T 5 --rho 0.5 --time-effects --unbalanced 0.2 \
  --methods plugin,explicit,auto5 --workers 6 --truth-units 300000 --out "$OUT/mc_tvar_time_unbal.csv"
python sim_readingb.py --design tvar --reps 200 --n-units 4000 --T 5 --rho 0.5 --methods plugin,explicit \
  --workers 6 --truth-units 300000 --out "$OUT/mc_tvar_big.csv"
