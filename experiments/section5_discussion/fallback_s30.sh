#!/bin/bash
# The fallback at s = 30 on the random cubic graphs of the Divide and Conquer runs,
# for the comparison of Section 5. Validated up to 1,500 vertices; beyond that the
# construction gives the side without building the whole embedding.
# (--out is relative to results/section4_fallback, where run_fallback.py writes)
cd "$(dirname "$0")"

# graph seed 1
for n in 100 150 200 250 300 400 500 600 800 1000; do
  python -B ../../src/fallback/run_fallback.py --family random_cubic --n $n --s 30 --workers 4 --validate --out ../section5_discussion/fallback_s30.csv
done
for n in 700 900 1100 1200 1300 1400 1500; do
  python -B ../../src/fallback/run_fallback.py --family random_cubic --n $n --s 30 --workers 4 --validate --out ../section5_discussion/fallback_s30.csv
done
for n in 2000 3000 5000 10000; do
  python -B ../../src/fallback/run_fallback.py --family random_cubic --n $n --s 30 --workers 4 --out ../section5_discussion/fallback_s30.csv
done

# graph seeds 2 to 10
for seed in 2 3 4 5 6 7 8 9 10; do
  for n in 100 200 300 400 500 600 700 800 900 1000; do
    python -B ../../src/fallback/run_fallback.py --family random_cubic --n $n --seed $seed --s 30 --workers 4 --validate --out ../section5_discussion/fallback_s30.csv
  done
done

# the largest graphs; a run already in the csv is skipped
done_already() {
  grep -q "^random_cubic,$1,$2,[0-9]*,[0-9]*,[0-9.]*,30," ../../results/section5_discussion/fallback_s30.csv
}
run() {
  done_already $1 $2 || python -B ../../src/fallback/run_fallback.py --family random_cubic --n $1 --seed $2 --s 30 --workers 8 --out ../section5_discussion/fallback_s30.csv
}
for n in 30000 100000; do
  run $n 1
done
for seed in 2 3 4 5 6 7 8 9 10; do
  for n in 2000 3000 5000 10000; do
    run $n $seed
  done
done
