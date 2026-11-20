#!/usr/bin/env python3
"""Seeded-bug audit of the selective suppression of lift-up and Orr (selective_lu_orr.md, section 5.5).

    split_bug_audit.py [--jobs 4] [--only B3,S1] [--keep] [--budget-only | --kelvin-only]
(--only takes names or prefixes: none, B1 (B1 and B10-B16), B10, W, U ...)

For each bug: the sources are copied to a scratch tree, the bug is planted by an exact text
replacement (the script stops if the text is not there: the sources moved on), the CPU build is made
there, and

  * the Kelvin decks tests/decks/kelvin_split_*.in are run (test_kelvin_s2, one rank); the margin of a
    deck is the largest of its errors (v, eta, energy, projection residual) divided by its tolerance,
    and for the order decks the observed order of the finest pair is listed too (required: 2.8);
  * for the bugs in the statistics, the budget run of tests/decks/split_budget_ny64.in is checked with
    tests/split_budget.py: the margin is the largest of (value / limit) over its checks;
  * for the bugs of the switch times (section 9 of the specification) that only a run of hst shows, the budget
    run of tests/decks/split_budget_switch.in with the same checks plus the times at which the log says the
    weights changed (limit 1e-8);
  * for the bugs that only the two small unit tests see (test_production_switch, test_split_stats) the test is run
    and must fail.

A bug is robust if some deck has a margin >= 10 (the solver without the bug has margins well below 1:
the row 'none' is the headroom).  Needs mpifort, FFTW, numpy; a CPU build per bug (about 15 s) and the
runs (about 1 min with --jobs 4).  The GPU build is not covered: a bug of the device mapping is checked
by running the suite with it (see selective_lu_orr.md).
"""
import argparse
import concurrent.futures as cf
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

EQ = 'src/hst_equations.f90'
PR = 'src/hst_production.f90'
ST = 'src/hst_stats.f90'

# name: (description, [(file, old, new), ...])
KELVIN_BUGS = {
    'B1_swap_weights': ('glu and gorr swapped', [
        (EQ, 'glc = glu(iz, ix)', 'glc = gorr(iz, ix)'), (EQ, 'goc = gorr(iz, ix)', 'goc = glu(iz, ix)')]),
    'B2_orr_not_complement': ('(1 - gorr) written as gorr', [(EQ, '- (1.0d0 - goc)*', '- goc*')]),
    'B3_extra_not_stored': ('extra d2v term not stored in oldrhs', [
        (EQ, 'oldrhs(iy, iz, ix, 2) = expl\n          ! eta equation', 'oldrhs(iy, iz, ix, 2) = -g*unkn\n          ! eta equation')]),
    'B4_d0_for_d1': ('der(iy, 0, j) instead of der(iy, 1, j) in the extra term', [
        (EQ, 'dv = dv + der(iy, 1, j)*V(iy + j, iz, ix, 2)', 'dv = dv + der(iy, 0, j)*V(iy + j, iz, ix, 2)')]),
    'B5_no_s2_in_extra': ('s2now missing in the extra d2v term', [
        (EQ, '(S*ialfa(ix) + s2now*ibeta(iz))*dv', '(S*ialfa(ix))*dv')]),
    'B6_no_s2_in_liftup': ('s2now missing in the eta source', [
        (EQ, 'glc*(s2now*ialfa(ix) - S*ibeta(iz))', 'glc*(-S*ibeta(iz))')]),
    'B7_glu_without_G': ('glu = prod_lu (the filter G left out)', [(PR, 'glu = prod_lu*gprod', 'glu = prod_lu')]),
    'B8_gorr_without_G': ('gorr = prod_orr (the filter G left out)', [(PR, 'gorr = prod_orr*gprod', 'gorr = prod_orr')]),
    'B9_eta_source_uses_gorr': ('the eta source uses gorr (the extra d2v term keeps its own weight)', [(EQ, 'glc = glu(iz, ix)', 'glc = gorr(iz, ix)')]),
}
SW = 'src/hst_production.f90'
HST = 'src/hst.f90'
# the switch times (prod_lu_start, prod_orr_start): seen by the Kelvin decks kelvin_split_sw_* and kelvin_split_order_sw_*
KELVIN_BUGS.update({
    'B10_switch_ignored': ('the switch times are ignored (the weights are prod_lu G from the start)', [
        (SW, 'in_effect = t >= ts - 1.0d-9*max(1.0d0, abs(ts))', 'in_effect = .true.')]),
    'B11_switch_inverted': ('the weights are prod_lu G before the switch time and G after it', [
        (SW, 'in_effect = t >= ts - 1.0d-9*max(1.0d0, abs(ts))', 'in_effect = t < ts - 1.0d-9*max(1.0d0, abs(ts))')]),
    'B12_start_times_swapped': ('the lift-up follows prod_orr_start', [
        (SW, 'lu_new = in_effect(t, prod_lu_start)', 'lu_new = in_effect(t, prod_orr_start)')]),
    'B13_update_at_step_end': ('the weights are updated at the end of the step instead of at its start', [
        (EQ, '    call update_production_weights(time)      ! the weights of the production for the whole step (switch times: hst_production)\n', ''),
        (EQ, '  end subroutine timestep', '    call update_production_weights(time)\n  end subroutine timestep')]),
    'B14_no_update_in_step': ('timestep does not update the weights (only outstats and the set-up do)', [
        (EQ, '    call update_production_weights(time)      ! the weights of the production for the whole step (switch times: hst_production)\n', '')]),
    'B15_before_weight_one': ('before the switch time the weight is 1, not G', [
        (SW, '    glu = merge(prod_lu, 1.0d0, lu_in)*gprod', '    glu = prod_lu*gprod\n    if (.not. lu_in) glu = 1.0d0')]),
    'B16_tolerance_none': ('a switch time is reached only for t >= ts exactly (no round-off tolerance)', [
        (SW, 'in_effect = t >= ts - 1.0d-9*max(1.0d0, abs(ts))', 'in_effect = t >= ts')]),
})
# seen by the budget run with switch times off the step grid (the log of hst and the statistics)
SWITCH_RUN_BUGS = {
    'W1_clip_not_wired': ('hst does not cut its steps to end on the switch times', [
        (HST, '    deltat = production_clip_dt(time, deltat)', '    deltat = deltat')]),
    'W2_clip_one_time': ('the cut ignores prod_orr_start', [
        (SW, '    d = clipped(prod_orr_start)\n', '')]),
}
# seen by the unit tests only: name: (description, edits, test program)
UNIT_BUGS = {
    'U1_clip_never_cuts': ('production_clip_dt never cuts', [
        (SW, 'if (t + d > ts - 1.0d-9*max(1.0d0, abs(ts))) clipped = ts - t', 'if (.false.) clipped = ts - t')], 'test_production_switch'),
    'U2_clip_cuts_reached': ('a switch time already reached still cuts the step (to a step of length 0)', [
        (SW, '      if (in_effect(t, ts)) return\n      if (t + d', '      if (t + d')], 'test_production_switch'),
    'U3_clip_tolerance': ('the cut leaves the step ending 1e-8 short of the switch time', [
        (SW, 'clipped = ts - t', 'clipped = ts - t - 1.0d-8')], 'test_production_switch'),
    'U4_stats_no_update': ('outstats does not bring the weights up to date', [
        (ST, '    call update_production_weights(time)       ! glu, gorr as in effect at this time (a row at a switch time shows the new ones)\n', '')],
        'test_split_stats'),
}
BUDGET_BUGS = {
    'S1_wrong_weight_in_stat': ('gP_LU summed with gorr', [(ST, 'glup = glup + glu(iz, ix)*pl', 'glup = glup + gorr(iz, ix)*pl')]),
    'S2_porr_sign': ('P_Orr with the wrong sign', [(ST, 'po = -w*aa/kh2*', 'po = w*aa/kh2*')]),
    'S3_no_E00': ('the (0,0) column left out of E_tor', [(ST, 'sp(1) = sp(1) + sp(9) ', 'sp(1) = sp(1) + 0*sp(9) ')]),
    'S4_plu_uses_a': ('P_LU with a instead of b', [(ST, 'pl = -w*bb/kh2*', 'pl = -w*aa/kh2*')]),
}


def sh(cmd, cwd, env=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, env=env, shell=isinstance(cmd, str), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if check and r.returncode != 0:
        raise RuntimeError('%s failed:\n%s' % (cmd, r.stdout[-3000:]))
    return r


def build(bug, edits, root):
    tree = tempfile.mkdtemp(prefix='bug_', dir=root)
    for d in ('src', 'tests'):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(tree, d), ignore=shutil.ignore_patterns('__pycache__', '*.csv', '*.fld', 'reference', 'decks'))
    shutil.copy(os.path.join(ROOT, 'Makefile'), tree)
    for f, old, new in edits:
        path = os.path.join(tree, f)
        s = open(path).read()
        if s.count(old) != 1:
            raise SystemExit('%s: the text to replace occurs %d times in %s:\n%s' % (bug, s.count(old), f, old))
        open(path, 'w').write(s.replace(old, new))
    sh('make -j8 all test BUILD=build', tree)
    return tree


def run_deck(tree, deck):
    work = tempfile.mkdtemp(prefix='run_', dir=tree)
    r = sh(['mpirun', '-np', '1', os.path.join(tree, 'build', 'test_kelvin_s2'), os.path.join(ROOT, 'tests', 'decks', deck)], work, check=False)
    out = r.stdout
    tol = float(re.search(r'tol\s*=\s*([0-9.eE+-]+)', open(os.path.join(ROOT, 'tests', 'decks', deck)).read()).group(1))
    rows = re.findall(r'err v\s+(\S+)\s+eta\s+(\S+)\s+energy\s+(\S+)\s+proj\s+(\S+)', out)
    if not rows:
        return dict(deck=deck, margin=float('inf'), quantity='crash', order=None, passed=False)
    worst, q = 0.0, ''
    for row in rows:
        for name, val in zip(('v', 'eta', 'energy', 'proj'), row):
            if float(val) / tol > worst:
                worst, q = float(val) / tol, name
    order = re.findall(r'observed order in \w+:\s+(\S+)', out)
    return dict(deck=deck, margin=worst, quantity=q, order=float(order[-1]) if order else None, passed='PASSED' in out and 'FAILED' not in out)


def kelvin_audit(tree, jobs):
    decks = sorted(f for f in os.listdir(os.path.join(ROOT, 'tests', 'decks')) if f.startswith('kelvin_split_'))
    with cf.ThreadPoolExecutor(jobs) as ex:
        return list(ex.map(lambda d: run_deck(tree, d), decks))


SWITCHES = (0.2037, 0.4511)          # split_budget_switch.in


def budget_audit(tree, deck='split_budget_ny64.in', switches=()):
    spec = importlib.util.spec_from_file_location('split_budget', os.path.join(HERE, 'split_budget.py'))
    sb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sb)
    work = tempfile.mkdtemp(prefix='budget_', dir=tree)
    with open(os.path.join(work, 'run.log'), 'w') as log:
        r = subprocess.run(['mpirun', '-np', '1', os.path.join(tree, 'build', 'hst'), os.path.join(ROOT, 'tests', 'decks', deck)],
                           cwd=work, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        return float('inf'), 'crash'
    r = sb.budget(work, switches)
    checks = {'E_tor+E_pol': (r['energy'], 1e-11), 'P_LU+P_Orr': (r['prod'], 1e-11), 'budget': (r['rel'], 2e-3)}
    if switches:
        got, want = sb.logged_switches(work), sorted(switches)
        checks['switch times'] = (max([abs(g - w) for g, w in zip(got, want)] + [0.0]) + (0.0 if len(got) == len(want) else 1.0), 1e-8)
    q, (val, lim) = max(checks.items(), key=lambda kv: kv[1][0] / kv[1][1])
    return val / lim, q


def unit_audit(tree, exe):
    work = tempfile.mkdtemp(prefix='unit_', dir=tree)
    # a test that does not end (a planted bug can make a loop of the test stand still) has failed
    r = sh(['timeout', '120', 'mpirun', '-np', '1', os.path.join(tree, 'build', exe), os.path.join(ROOT, 'tests', 'decks', 'pressure.in')], work, check=False)
    return r.returncode != 0 or 'PASSED' not in r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--only', default='', help='comma-separated bug names or prefixes (none, B3, S1, ...)')
    ap.add_argument('--keep', action='store_true', help='keep the scratch trees')
    ap.add_argument('--budget-only', action='store_true')
    ap.add_argument('--kelvin-only', action='store_true')
    a = ap.parse_args()
    only = [x for x in a.only.split(',') if x]
    pick = lambda n: not only or any(n.startswith(o) for o in only)

    root = tempfile.mkdtemp(prefix='split_audit_')
    status = 0
    print('| bug | what | detecting deck | quantity | margin | order (finest pair) | verdict |')
    print('| --- | --- | --- | --- | --- | --- | --- |')
    try:
        todo = [('none', 'the solver as it is', [])] if pick('none') else []
        if not a.budget_only:
            todo += [(n, d, e) for n, (d, e) in KELVIN_BUGS.items() if pick(n)]
        for name, desc, edits in todo:
            tree = build(name, edits, root)
            res = kelvin_audit(tree, a.jobs)
            best = max(res, key=lambda r: r['margin'])
            orders = [r['order'] for r in res if r['order'] is not None]
            # the order decks fail on their own criterion (finest pair >= 2.8) without an amplitude margin
            failing = [r for r in res if not r['passed']]
            ostr = ', '.join('%.2f' % o for o in orders)
            if name == 'none':
                verdict = 'all pass' if not failing else 'FAILS: ' + ', '.join(r['deck'] for r in failing)
                status |= bool(failing)
                print('| %s | %s | %s | %s | %.2f (headroom %.0fx) | %s | %s |' % (name, desc, best['deck'], best['quantity'], best['margin'], 1 / best['margin'], ostr, verdict))
            else:
                verdict = 'robust' if best['margin'] >= 10 else ('caught, fragile' if failing else 'MISSED')
                if best['margin'] < 10:
                    status |= 1
                print('| %s | %s | %s | %s | %.1f | %s | %s (%d of %d decks fail) |' % (name, desc, best['deck'], best['quantity'], best['margin'], ostr, verdict, len(failing), len(res)))
            sys.stdout.flush()
        if not a.kelvin_only:
            for name, (desc, edits) in SWITCH_RUN_BUGS.items():
                if not pick(name):
                    continue
                tree = build(name, edits, root)
                margin, q = budget_audit(tree, 'split_budget_switch.in', SWITCHES)
                verdict = 'robust' if margin >= 10 else ('caught, fragile' if margin >= 1 else 'MISSED')
                if margin < 10:
                    status |= 1
                print('| %s | %s | split_budget_switch | %s | %.1f | | %s |' % (name, desc, q, margin, verdict))
                sys.stdout.flush()
            for name, (desc, edits, exe) in UNIT_BUGS.items():
                if not pick(name):
                    continue
                tree = build(name, edits, root)
                failed = unit_audit(tree, exe)
                if not failed:
                    status |= 1
                print('| %s | %s | %s | fails | | | %s |' % (name, desc, exe, 'robust (the test fails)' if failed else 'MISSED'))
                sys.stdout.flush()
            for name, (desc, edits) in BUDGET_BUGS.items():
                if not pick(name):
                    continue
                tree = build(name, edits, root)
                margin, q = budget_audit(tree)
                verdict = 'robust' if margin >= 10 else ('caught, fragile' if margin >= 1 else 'MISSED')
                if margin < 10:
                    status |= 1
                print('| %s | %s | split_budget_ny64 | %s | %.1f | | %s |' % (name, desc, q, margin, verdict))
                sys.stdout.flush()
    finally:
        if not a.keep:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print('scratch trees kept in', root)
    sys.exit(status)


if __name__ == '__main__':
    main()
