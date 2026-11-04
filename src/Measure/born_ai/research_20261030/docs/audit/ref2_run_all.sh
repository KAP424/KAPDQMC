#!/bin/bash
# Re-runs all referee scripts and concatenates their stdout into ref2_tables.txt (numbers only, no prose).
cd "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit"
OUT=ref2_tables.txt
: > $OUT
for s in ref2_I_power ref2_I_drift ref2_I_blind ref2_I_excess ref2_I_delta ref2_I_plateau ref2_I_contours ref2_I_interp ref2_I_shift3 ref2_I_shift4 ref2_II_identity ref2_II_honeycomb ref2_II_nodes2 ref2_II_nodes3 ref2_II_data ref2_II_L3 ref2_III; do
  echo "=================== $s ===================" >> $OUT
  D:/app/conda/python.exe -B -I $s.py >> $OUT 2>&1
done
echo "ALL DONE" >> $OUT
