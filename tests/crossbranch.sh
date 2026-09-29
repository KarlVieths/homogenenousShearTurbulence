#!/bin/bash
# Restart round trip: 25 steps of the small deck with build A at npy A, the
# other 25 with build B at npy B, restarted from A's Dati.cart.out, the final
# field compared with tests/reference/small.fld at 1e-10 (the restart file is
# exact, so the tolerance is that of the regression).  The two builds may be
# the same one (then it is the restart test between npy values, e.g. 1 -> 2
# and 2 -> 1) or of different checkouts (it began as the round trip between
# main and the branch multinode-y, merged in session 11): the same deck and
# the same field files must work on both.
#   tests/crossbranch.sh <build A> <build B> [nranks] [npy A] [npy B]
# The builds are directories.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
A=$(cd "$1" && pwd); B=$(cd "$2" && pwd); np=${3:-2}; npyA=${4:-1}; npyB=${5:-1}
work=$(mktemp -d); cd "$work" || exit 1
run() {   # run <build> <npy> <time_from_restart>
  sed -e "s/&mesh /\&mesh npy = $2, /" -e "s/nstep = [0-9]*/nstep = 25/" \
      -e "s/time_from_restart = .false./time_from_restart = $3/" "$here/tests/decks/small.in" > hst.in
  echo "--- $1/hst on $np ranks, npy = $2, restart = $3"
  mpirun -np "$np" "$1/hst" hst.in > "run$3.log" 2>&1 || { echo "run failed, see $work/run$3.log"; exit 1; }
  grep -E "ranks =|end of run" "run$3.log"
}
run "$A" "$npyA" .false.
run "$B" "$npyB" .true.
python3 "$here/tests/compare_fields.py" Dati.cart.out "$here/tests/reference/small.fld" 1e-10 | tail -2
status=${PIPESTATUS[0]}
rm -rf "$work"
[ "$status" -eq 0 ] && echo "ROUND TRIP OK" || echo "ROUND TRIP FAILED"
exit "$status"
