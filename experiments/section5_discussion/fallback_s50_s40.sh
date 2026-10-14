#!/bin/bash
# The fallback at s = 50 and 40 on random cubic graphs of 1,000 to 30,000 vertices,
# for the comparison of Section 5.1 of the report, validated at 1,000 vertices.
# (--out is relative to results/section4_fallback, where run_fallback.py writes)
S="$(cd "$(dirname "$0")" && pwd)"
cd "$S"
for s in 50 40; do
  for n in 1000 3000 10000 30000; do
    v=""; [ $n -eq 1000 ] && v="--validate"
    python -B ../../src/fallback/run_fallback.py --family random_cubic --n $n --s $s --workers 12 $v --out ../section5_discussion/fallback_s50_s40.csv
  done
done
echo "SCALE DONE"
