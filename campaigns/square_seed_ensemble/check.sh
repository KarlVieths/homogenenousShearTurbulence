#!/bin/bash
# The launch checks of the ensemble (README, "Before launching").
#   ./check.sh decks          the decks differ in the seed line only (the header line names the run)
#   ./check.sh logs [test]    per run: the seed the log reports, the first q2 of Runtimedata, s/step;
#                             the seeds must be four distinct numbers, the first q2 about 0.1 in all
cd "$(dirname "$(readlink -f "$0")")"
case "${1:-}" in
  decks)
    first=$(ls -d runs/*/ | head -n1); bad=0
    for d in runs/*/; do
      [ "$d" = "$first" ] && continue
      echo "--- $first  vs  $d  (lines other than the header line)"
      diff <(tail -n +2 $first/hst.in) <(tail -n +2 $d/hst.in) | grep '^[<>]'
      [ "$(diff <(tail -n +2 $first/hst.in) <(tail -n +2 $d/hst.in) | grep -c '^[<>]')" -eq 2 ] || bad=1
    done
    [ $bad -eq 0 ] && echo "OK: only the seed differs" || { echo "NOT OK: more than the seed differs"; exit 1; }
    ;;
  logs)
    base=${2:-runs}
    printf "%-22s %8s %9s %9s\n" run seed "first q2" "s/step"
    for d in $base/*/; do
      lastlog=$(ls -t $d/run-*.log 2>/dev/null | head -1)
      seed=-; [ -n "$lastlog" ] && seed=$(sed -n 's/.*generating random initial field: seed = *\([0-9]*\).*/\1/p' $lastlog | head -1)
      ly=$(sed -n 's/^ *ly *= *\([0-9.eE+-]*\).*/\1/p' $d/hst.in | head -1)
      q2=-; [ -s $d/statistics/Runtimedata ] && q2=$(head -n1 $d/statistics/Runtimedata | awk -v ly=$ly '{printf "%.4f", 2*$10/ly}')
      sps=-; [ -n "$lastlog" ] && sps=$(grep -o '[0-9.]* s/step' $lastlog | tail -1 | cut -d' ' -f1)
      printf "%-22s %8s %9s %9s\n" $(basename $d) "${seed:--}" $q2 "${sps:--}"
    done
    ;;
  *) sed -n '2,5p' "$0";;
esac
