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
run "$np" test_pressure pressure.in
run "$np" test_taylorgreen pressure.in
run 1     test_forcing pressure.in
run "$np" test_conservation pressure.in
run "$np" test_stokes stokes.in
run "$np" test_stats pressure.in
rm -rf "$work"
[ $status -eq 0 ] && echo "ALL PASSED" || echo "SOME FAILED"
exit $status
