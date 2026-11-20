#!/bin/bash
# Run the test programs of a build.   usage:  tests/run_tests.sh [build-cpu|build-gpu] [nranks] [npy]
# Each test prints PASSED or FAILED; the script exits non-zero if any failed.
# With npy > 1 the multi-rank runs put that many y slabs into the deck
# (nranks must be a multiple of npy).
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
build=${1:-build-cpu}
np=${2:-1}
npy=${3:-1}
work=$(mktemp -d)
cd "$work"
status=0
run() {   # run <ranks> <exe> <deck>
  deck="$here/tests/decks/$3"; note=""
  if [ "$1" -gt 1 ] && [ "$npy" -gt 1 ]; then
    sed "s/&mesh /\&mesh npy = $npy, /" "$deck" > "$work/$3"; deck="$work/$3"; note=", npy = $npy"
  fi
  echo "--- $2 on $1 rank(s), deck $3$note"
  mpirun -np "$1" "$here/$build/$2" "$deck" 2>&1 | grep -v -E "Authorization|hcoll|^ *$"
  [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
}
run "$np" test_roundtrip small.in
run "$np" test_linsolve small.in
run 1     test_kelvin kelvin.in
run 1     test_kelvin kelvin_exact.in
run 1     test_kelvin kelvin_s2const.in
run 1     test_kelvin kelvin_s2osc.in
run 1     test_kelvin kelvin_s2mean.in
run 1     test_kelvin kelvin_stretched.in
run 1     test_kelvin kelvin_damped.in
run 1     test_kelvin kelvin_hard.in
run 1     test_kelvin kelvin_outside.in
run 1     test_kelvin kelvin_unforced.in
run 1     test_kelvin kelvin_unforced_s2.in
run 1     test_kelvin kelvin_halfforced.in
run 1     test_kelvin kelvin_forced.in
# Kelvin modes in an oscillating S2 (tests/kelvin_test_unsteady_S2.md): the time
# series of one column against its closed form, with the production filter, and
# the observed order in time; four of them also write their table, which
# kelvin_ref.py (an independent implementation) checks.
run_s2() {   # run_s2 <deck> [x]: with x, also kelvin_ref.py --compare on the table
  run 1 test_kelvin_s2 "$1"
  if [ -n "${2:-}" ]; then
    if python3 -c "import numpy, scipy" 2>/dev/null; then
      python3 "$here/tests/kelvin_ref.py" --compare "$work/${1%.in}.csv" --tol 1e-5 || status=1
    else
      echo "    (kelvin_ref.py skipped: numpy or scipy not available)"
    fi
  fi
}
run_s2 kelvin_s2_a_G0.in x
run_s2 kelvin_s2_a_G1.in x
run_s2 kelvin_s2_a_Gh_fast.in
run_s2 kelvin_s2_a_G0_slow.in
run_s2 kelvin_s2_a_G0_inviscid.in
run_s2 kelvin_s2_b_G0.in x
run_s2 kelvin_s2_b_G1.in
run_s2 kelvin_s2_b_Gh.in
run_s2 kelvin_s2_b_G0_fast.in
run_s2 kelvin_s2_b_Gh_start.in x
run_s2 kelvin_s2_order_a_G1.in
run_s2 kelvin_s2_order_b_Gh.in
run_s2 kelvin_s2_order_a_G1_const.in
# The two mechanisms of the production, the lift-up (prod_lu) and the Orr energy transfer
# (prod_orr), weighted apart (selective_lu_orr.md, section 5): Kelvin modes of both kinds in a
# constant and in an oscillating S2, the weights times a filter G, weights outside [0, 1], both
# weights 0, the order of accuracy (finest pair >= 2.8, nu = 1e-12) in v without the Orr transfer
# and in eta with both, and four tables checked by kelvin_ref.py.
for m in a b; do
  for w in lu0orr1 lu1orr0 mix; do
    for s in osc const; do
      x=; [ "$w" = mix ] && [ "$s" = osc ] && x=x
      run_s2 kelvin_split_${m}_${w}_${s}.in $x
    done
  done
done
run_s2 kelvin_split_a_mix_filter.in x
run_s2 kelvin_split_b_mix_filter.in x
run_s2 kelvin_split_b_outside.in
run_s2 kelvin_split_b_zero_osc.in
run_s2 kelvin_split_order_a_orr0.in
run_s2 kelvin_split_order_b_eta.in
run_s2 kelvin_split_order_b_eta_g547.in
# The energy budget of a nonlinear run with the weights, from statistics/split_runtime.dat and
# Runtimedata (tests/split_budget.py): the identities to round-off, the budget to the residual of the
# discretisation, which must fall with ny.  Each run in a directory of its own (the statistics
# files are appended to).
run_budget() {   # run_budget <ranks>
  if ! python3 -c "import numpy" 2>/dev/null; then echo "--- split budget skipped: numpy not available"; return; fi
  for ny in 64 128; do
    mkdir -p "$work/budget$ny"; deck="$here/tests/decks/split_budget_ny$ny.in"; note=""
    if [ "$1" -gt 1 ] && [ "$npy" -gt 1 ]; then
      sed "s/&mesh /\&mesh npy = $npy, /" "$deck" > "$work/budget$ny/hst.in"; deck="$work/budget$ny/hst.in"; note=", npy = $npy"
    fi
    echo "--- hst on $1 rank(s), deck split_budget_ny$ny.in$note"
    ( cd "$work/budget$ny" && mpirun -np "$1" "$here/$build/hst" "$deck" > run.log 2>&1 ) || { echo "run failed, see $work/budget$ny/run.log"; status=1; return; }
  done
  echo "--- split_budget.py: ny = 128 (and the fall of the residual from ny = 64)"
  python3 "$here/tests/split_budget.py" "$work/budget128" --margin 20 --tol 5e-4 --prev "$work/budget64" --ratio 3 || status=1
  echo "--- split_budget.py: ny = 64"
  python3 "$here/tests/split_budget.py" "$work/budget64" --margin 20 --tol 2e-3 || status=1
}
run_budget "$np"
run "$np" test_pressure pressure.in
run "$np" test_taylorgreen pressure.in
run 1     test_forcing pressure.in
run "$np" test_conservation pressure.in
run "$np" test_stokes stokes.in
run "$np" test_stats pressure.in
run "$np" test_split_stats pressure.in      # split_runtime.dat against a reference summed the other way round (weights, k_par = 0, k_perp = 0)
rm -rf "$work"
[ $status -eq 0 ] && echo "ALL PASSED" || echo "SOME FAILED"
exit $status
