#!/bin/bash
# One line per run: state, simulation time reached, s/step of the last segment.
#   ./status.sh          the runs in runs/
#   ./status.sh test     the test runs in test/
cd "$(dirname "$(readlink -f "$0")")"
base=${1:-runs}
printf "%-22s %-11s %10s %9s  %s\n" run state "S t" "s/step" "last log"
for d in $base/*; do
  [ -d $d ] || continue
  state=new
  [ -f $d/Dati.cart.out ] && state=started
  for s in DONE INCOMPLETE FAILED; do [ -f $d/$s ] && state=$s; done
  t=-; [ -s $d/statistics/Runtimedata ] && t=$(tail -n1 $d/statistics/Runtimedata | awk '{printf "%.1f", $1}')
  lastlog=$(ls -t $d/run-*.log 2>/dev/null | head -1)
  sps=-; [ -n "$lastlog" ] && sps=$(grep -o '[0-9.]* s/step' $lastlog | tail -1 | cut -d' ' -f1)
  printf "%-22s %-11s %10s %9s  %s\n" ${d#$base/} $state $t "${sps:--}" "$(basename "$lastlog" 2>/dev/null)"
done
