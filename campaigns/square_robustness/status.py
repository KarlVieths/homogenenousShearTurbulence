#!/usr/bin/env python3
"""One line per run: state, simulation time, s/step, the current q2 and its trend, the laminar check.

    ./status.py                    # every run of the manifest that has a directory in runs/
    ./status.py --stages 3a        # only these stages
    ./status.py --base test        # the test runs of ./submit.py --test
    ./status.py --baseline-q2 0.1  # q2 of the baseline (default: the mean of ../square_seed_ensemble/seed_ensemble.csv)
    ./status.py --mem              # GPU memory of the jobs (logs/gpumem-*.log): start, last sample, growth per hour
    ./status.py --stop-laminar     # touch STOP in the stage-3a runs that the check flags (the job ends those runs, README)

state: new (no restart file), started, DONE (t_max reached), INCOMPLETE (stopped by the time limit, a chained
job continues it), STOPPED (STOP file), FAILED.  q2 is the mean over the last 50 time units of statistics/Runtimedata,
d(q2) the change against the 50 before.  The laminar check (stage 3a, after t = 400): q2 below 1 % of the baseline and
still falling -> flagged "LAMINAR: touch STOP".
"""
import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def tail_lines(path, nbytes=4_000_000):
    with open(path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        data = f.read().decode(errors='replace').splitlines()
    return data[1:] if size > nbytes else data


def read_tail(path):
    """(t, q2) arrays of the last rows of Runtimedata; q2 = 2 energy/ly needs ly from the deck, so return the raw energy."""
    rows = []
    for l in tail_lines(path):
        p = l.split()
        if len(p) < 13:
            continue
        try:
            rows.append((float(p[0]), float(p[9])))
        except ValueError:
            # Fortran writes |x| < 1e-99 as 1.234-100: skip such a row
            continue
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default='runs')
    ap.add_argument('--stages', default='')
    ap.add_argument('--manifest', default='manifest.csv')
    ap.add_argument('--baseline-q2', type=float, default=0.0)
    ap.add_argument('--stop-laminar', action='store_true')
    ap.add_argument('--mem', action='store_true', help='summarise the GPU-memory logs of the jobs (a leak shows as growth)')
    a = ap.parse_args()
    if a.mem:
        import glob
        print(f"{'job log':32s} {'gpu':5s} {'samples':>7s} {'first MiB':>10s} {'last MiB':>9s} {'max MiB':>8s} {'growth MiB/h':>13s}")
        for f in sorted(glob.glob(os.path.join(HERE, 'logs', 'gpumem-*.log'))):
            series = {}
            for l in open(f):
                p = l.split()
                try:
                    el = float(p[0])
                except (ValueError, IndexError):
                    continue
                for kv in p[1:]:
                    if '=' in kv:
                        k, v = kv.split('=')
                        try:
                            series.setdefault(k, []).append((el, float(v)))
                        except ValueError:
                            pass
            for k, s in sorted(series.items()):
                grow = (s[-1][1] - s[0][1])/((s[-1][0] - s[0][0])/3600.0) if len(s) > 1 and s[-1][0] > s[0][0] else float('nan')
                print(f"{os.path.basename(f):32s} {k:5s} {len(s):7d} {s[0][1]:10.0f} {s[-1][1]:9.0f} {max(v for _, v in s):8.0f} {grow:13.1f}")
        return
    stages = [s for s in a.stages.split(',') if s]
    q2b = a.baseline_q2
    seeds = os.path.join(HERE, '..', 'square_seed_ensemble', 'seed_ensemble.csv')
    if not q2b and os.path.exists(seeds):
        for r in csv.DictReader(open(seeds)):
            if r['quantity'] == 'q2':
                q2b = float(r['mean'])
    rows = list(csv.DictReader(open(os.path.join(HERE, a.manifest))))
    print(f"{'run':20s} {'stage':5s} {'state':10s} {'t':>7s} {'t_max':>6s} {'s/step':>7s} {'q2':>8s} {'d(q2)':>8s} {'q2/base':>8s}  note")
    flagged = []
    for r in rows:
        if stages and r['stage'] not in stages:
            continue
        d = os.path.join(HERE, a.base, r['id'])
        if not os.path.isdir(d):
            continue
        state = 'new'
        if os.path.exists(os.path.join(d, 'Dati.cart.out')):
            state = 'started'
        for s in ('DONE', 'INCOMPLETE', 'STOPPED', 'FAILED'):
            if os.path.exists(os.path.join(d, s)):
                state = s
        tmax = ''
        m = re.search(r'^\s*t_max\s*=\s*([0-9.eE+-]+)', open(os.path.join(d, 'hst.in')).read(), re.M)
        if m:
            tmax = '%g' % float(m.group(1))
        ly = float(re.search(r'^\s*ly\s*=\s*([0-9.eE+-]+)', open(os.path.join(d, 'hst.in')).read(), re.M).group(1))
        logs = sorted([f for f in os.listdir(d) if f.startswith('run-') and f.endswith('.log')],
                      key=lambda f: os.path.getmtime(os.path.join(d, f)))
        sps = '-'
        if logs:
            found = re.findall(r'([0-9.]+) s/step', open(os.path.join(d, logs[-1]), errors='replace').read()[-20000:])
            sps = found[-1] if found else '-'
        t = q2 = dq = ratio = '-'
        note = ''
        rt = os.path.join(d, 'statistics', 'Runtimedata')
        if os.path.exists(rt) and os.path.getsize(rt) > 0:
            data = read_tail(rt)
            if data:
                tt = data[-1][0]
                last = [e for (x, e) in data if x > tt - 50]
                prev = [e for (x, e) in data if tt - 100 < x <= tt - 50]
                qn = 2*sum(last)/len(last)/ly
                t = '%.1f' % tt
                q2 = '%.4f' % qn
                if prev:
                    qp = 2*sum(prev)/len(prev)/ly
                    dq = '%+.4f' % (qn - qp)
                    if r['stage'] == '3a' and q2b and tt > 450 and qn < 0.01*q2b and qn < qp and state != 'STOPPED':
                        note = 'LAMINAR: touch STOP'
                        flagged.append(r['id'])
                if q2b:
                    ratio = '%.3f' % (qn/q2b)
        print(f"{r['id']:20s} {r['stage']:5s} {state:10s} {t:>7s} {tmax:>6s} {sps:>7s} {q2:>8s} {dq:>8s} {ratio:>8s}  {note}")
    if flagged:
        if a.stop_laminar:
            for i in flagged:
                open(os.path.join(HERE, a.base, i, 'STOP'), 'w').close()
            print('\nSTOP written in:', ' '.join(flagged))
        else:
            print('\nflagged (stage 3a): ' + ' '.join(flagged) + '   ->  ./status.py --stop-laminar')


if __name__ == '__main__':
    main()
