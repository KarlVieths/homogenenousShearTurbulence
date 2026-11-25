#!/usr/bin/env python3
"""Plan and submit runs of the robustness campaign on HoreKA2.  Run it on a login node, from this directory.

    ./submit.py --stages 1a,1b --dry-run        # the plan: jobs, runs per job, time limits, sbatch commands; nothing submitted
    ./submit.py --stages 1a --test              # 8 minutes, decks copied to test/: speed, step, memory, launch (not gated)
    ./submit.py --stages 1a,1b,1c --ack stage0  # the real jobs
    ./submit.py --ids 3b-B-Rc6000-t00 --gpus 1 --ack stage0,stage1   # one GPU instead of the manifest's four (the H100 and H200 have 80 and 141 GB)

Needs HST_ACCOUNT and HST_PARTITION (or --account, --partition) for anything but --dry-run; HST_ROOT (default ~/hst) and
HST_TEST_PARTITION (a development partition for --test).  Single-GPU runs go four to a node-job (the job
script runs one per GPU; the runs of a job are chosen of similar length, as a job ends when its longest run does);
a run of G GPUs gets a job of its own (RUN_GPUS = G).  Every job asks for the whole node, unless --exact-gpus.
A job holds its GPUs until its longest run ends.  The plan prints what that costs under both ways of billing: node-hours
(the sum over the jobs of the longest run's wall hours at --speed) and GPU-hours (the GPUs each job holds), and the GPU-hours
of one job per run.  The documentation says GPU hours (`kit_project_usage`, partitions in "Shared" mode); if your project is billed
per GPU, `--pack 1 --exact-gpus` (one job per run, only its GPUs) is the cheap choice; if it is billed per node it costs up to four
times as much per run.  README, "Billing", says how to find out.

The time limit of a job is the longest run's est_wall_h (A100 hours of the manifest) / --speed x --margin-factor,
at most --max-time (2 days on the GPU partitions); a longer run continues in a chained job (the job script resubmits it).
--speed 1 is the A100 figure; the first H100 test of the 15 x 2 x 5 campaign was 1.58 times faster (0.079 -> 0.050 s/step), with an
unmeasured 1.0-1.6 range for the square-box grids: set it after ./submit.py --test (the node-hour estimate scales with it).

Gates (the checklist of the planning document, "Before launching"): a real submission needs --ack with the tokens
    stage0   stage 0 analysed (sigma_seed, R, seeds per point) and the open items of the seed campaign resolved   (all stages)
    stage1   stage 1 reviewed, the adopted box fixed, the manifest regenerated if it differs from the baseline   (1d, 2, 3a, 3b)
    nccl     the NCCL build rerun with the filter (Kelvin tests, regression) before a multi-GPU run              (runs with gpus > 1)
Every submission is appended to logs/submitted.tsv, and a run in it is not submitted again without --again
(two jobs in one run directory would corrupt it).
"""
import argparse
import csv
import datetime
import math
import os
import shutil
import subprocess
import sys

from make_manifest import EFF

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIRES = {'1a': ['stage0'], '1b': ['stage0'], '1c': ['stage0'], '1e': ['stage0'],
            '1d': ['stage0', 'stage1'], '2': ['stage0', 'stage1'], '3a': ['stage0', 'stage1'], '3b': ['stage0', 'stage1']}
GPU_MEM_GB = {'h100': 94, 'h200': 141, 'b200': 186}      # HoreKA2 hardware table: H100-94, H200, B200
CHECKLIST = """Checklist (planning document, "Before launching"):
  [stage0]  stage 0 analysed: sigma_seed and R per quantity, seeds per point for stages 1-2; the seed campaign's open items resolved
  [ -    ]  s/step, mean dt and peak GPU memory read from a run (./submit.py --test), est_wall_h recomputed (make_manifest.py constants)
  [ -    ]  mesh densities (16, 47.5, 24 per unit length) checked against the square-box deck
  [ nccl ]  the NCCL build rerun with the filter before a multi-GPU run
  [stage1]  stage 1 reviewed, the adopted box fixed before the rows of 1d, 2, 3 are used
  [ -    ]  workspace quota for the output (see the total above)"""


def fft_fit(n):
    while n % 2 == 0:
        n //= 2
    return n in (1, 3)


def nzd(nz):
    n = 3*nz
    while not fft_fit(n):
        n += 1
    return n


def hhmmss(hours):
    s = int(math.ceil(hours*3600/60.0))*60
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    return (f'{d}-' if d else '') + f'{h:02d}:{s//60:02d}:00'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stages', default='', help='comma-separated stages (1a,1b,1c,1d,1e,2,3a,3b)')
    ap.add_argument('--ids', nargs='*', default=[], help='run ids')
    ap.add_argument('--manifest', default='manifest.csv')
    ap.add_argument('--test', action='store_true', help='8-minute test in test/ (MARGIN 120 s, no chain; MAXSEG=2 in the environment: chain test)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--ack', default='', help='comma-separated gates: stage0, stage1, nccl')
    ap.add_argument('--again', action='store_true', help='submit runs that logs/submitted.tsv already lists')
    ap.add_argument('--account', default=os.environ.get('HST_ACCOUNT'))
    ap.add_argument('--partition', default=os.environ.get('HST_PARTITION'))
    ap.add_argument('--test-partition', default=os.environ.get('HST_TEST_PARTITION'))
    ap.add_argument('--node-gpus', type=int, default=4, help='GPUs per node of the partition (gpu-h200-8: 8)')
    ap.add_argument('--pack', type=int, default=0, help='runs per job (default: as many as fit the node)')
    ap.add_argument('--exact-gpus', action='store_true',
                    help='ask for the GPUs the runs need, not the whole node (if the partition shares nodes: --pack 1 --exact-gpus gives one job per run and no idle GPU)')
    ap.add_argument('--gpus', type=int, default=0, help='GPUs per run for every selected run (default: the manifest\'s)')
    ap.add_argument('--gpu-mem', type=float, default=0, help='GB per GPU (default from the partition name: h100 80, h200 141, b200 180)')
    ap.add_argument('--speed', type=float, default=1.0, help='speed of the GPUs relative to the A100 of est_wall_h')
    ap.add_argument('--margin-factor', type=float, default=1.25, help='time limit = longest est_wall_h / speed x this')
    ap.add_argument('--max-time', type=float, default=48.0, help='longest time limit in hours (the partition\'s)')
    ap.add_argument('--time', help='time limit of every job instead (e.g. 12:00:00)')
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(HERE, a.manifest))))
    stages = [s for s in a.stages.split(',') if s]
    unknown = [i for i in a.ids if i not in {r['id'] for r in rows}] + [s for s in stages if s not in REQUIRES]
    if unknown:
        sys.exit(f'unknown stage or id: {unknown}')
    sel = [r for r in rows if (r['stage'] in stages) or (r['id'] in a.ids)]
    if not sel:
        sys.exit('nothing selected: give --stages and/or --ids (see the docstring)')
    for r in sel:
        r['G'] = a.gpus or int(r['gpus'])
        if a.node_gpus % r['G']:
            sys.exit(f"{r['id']}: {r['G']} GPUs per run do not divide the node's {a.node_gpus}")
        if r['G'] > 1 and ((int(r['nx']) + 1) % r['G'] or nzd(int(r['nz'])) % r['G']):
            sys.exit(f"{r['id']}: {r['G']} ranks do not divide nx+1 = {int(r['nx']) + 1} and nzd = {nzd(int(r['nz']))}")

    part = (a.test_partition if a.test and a.test_partition else a.partition) or ''
    gmem = a.gpu_mem or next((v for k, v in GPU_MEM_GB.items() if k in part), 40.0)
    base = 'test' if a.test else 'runs'

    # memory: per GPU against the card
    for r in sel:
        per = float(r['est_mem_gb'])/r['G']
        if per > 0.9*gmem:
            sys.exit(f"{r['id']}: about {per:.0f} GB per GPU does not fit a {gmem:.0f} GB card: more GPUs per run (--gpus)")
        if r['G'] > 1 and float(r['est_mem_gb']) < 0.7*gmem:
            print(f"note: {r['id']} needs about {float(r['est_mem_gb']):.0f} GB and fits one {gmem:.0f} GB GPU: --gpus 1 avoids the multi-GPU run (and the nccl gate)")

    # gates
    need = set()
    for r in sel:
        need.update(REQUIRES[r['stage']])
        if r['G'] > 1:
            need.add('nccl')
    have = {x for x in a.ack.split(',') if x}
    missing = sorted(need - have)
    real = not a.test and not a.dry_run
    if real and missing:
        print(CHECKLIST)
        sys.exit(f'\nnot submitted: confirm {", ".join(missing)} with --ack {",".join(sorted(need))}')

    # decks, leftovers, earlier submissions
    log = os.path.join(HERE, 'logs', 'submitted.tsv')
    done_ids = {l.split('\t')[2] for l in open(log)} if os.path.exists(log) else set()
    keep = []
    for r in sel:
        d = os.path.join(HERE, 'runs', r['id'])
        if not os.path.exists(os.path.join(d, 'hst.in')):
            sys.exit(f"no deck {d}/hst.in: python3 make_runs.py")
        if real and r['id'] in done_ids and not a.again:
            print(f"skip {r['id']}: already in logs/submitted.tsv (--again to submit it anyway)"); continue
        if real and os.path.exists(os.path.join(d, 'Dati.cart.out')) and not [f for f in os.listdir(d) if f.startswith('run-')]:
            sys.exit(f"{d} has a Dati.cart.out but no run log: a leftover would replace the random field. Remove it.")
        keep.append(r)
    sel = keep
    if not sel:
        sys.exit('nothing to submit')

    # jobs: runs of the same GPU count, similar length together, node_gpus/G of them per job
    jobs = []
    for g in sorted({r['G'] for r in sel}):
        rs = sorted([r for r in sel if r['G'] == g], key=lambda r: -float(r['est_wall_h']))
        per = min(a.pack or a.node_gpus//g, a.node_gpus//g)
        for k in range(0, len(rs), per):
            jobs.append((g, rs[k:k + per]))
    totals = dict(nodeh=0.0, gpuh=0.0, useful=0.0, disk=0.0)   # node-hours and GPU-hours held by the jobs, node-hours if no GPU idled, output
    plan = []
    for g, rs in jobs:
        # a run of G GPUs on A100s estimated at est_wall_h wall hours (the manifest lists wall hours on its own gpus)
        wall = max(float(r['est_wall_h'])*int(r['gpus'])*EFF.get(int(r['gpus']), 0.6)/(g*EFF.get(g, 0.6)) for r in rs)/a.speed
        if a.test:
            limit, nseg = '00:08:00', 1
        elif a.time:
            limit, nseg = a.time, None
        else:
            hours = min(a.max_time, wall*a.margin_factor)
            limit = hhmmss(max(hours, 0.25))
            nseg = int(math.ceil(wall*a.margin_factor/a.max_time))
        name = 'hst-rob-' + ('test' if a.test else (rs[0]['stage'] if len({r['stage'] for r in rs}) == 1 else 'mix'))
        env = dict(os.environ, RUNS=' '.join(f'{base}/{r["id"]}' for r in rs), RUN_GPUS=str(g))
        if a.test:
            env.update(MARGIN=os.environ.get('MARGIN', '120'), MAXSEG=os.environ.get('MAXSEG', '1'))   # MAXSEG=2: the chain test (README)
        env['HST_ROOT'] = os.environ.get('HST_ROOT', os.path.expanduser('~/hst'))
        ng = len(rs)*g if a.exact_gpus else a.node_gpus
        cmd = ['sbatch', '--parsable', '--export=ALL', f'--job-name={name}', f'--gres=gpu:{ng}',
               f'--ntasks-per-node={ng}', f'--time={limit}']
        if a.account:
            cmd.append(f'--account={a.account}')
        if part:
            cmd.append(f'--partition={part}')
        cmd.append('horeka2_runs.slurm')
        plan.append((rs, g, wall, limit, nseg, cmd, env))
        # HoreKA2 charges node-hours: a job holds its node(s) until its longest run ends (a job of fewer GPUs, --exact-gpus,
        # is counted by its share of the node: only right if the partition shares nodes and charges that way)
        totals['nodeh'] += (8/60.0 if a.test else wall)          # one node per job, whatever the number of GPUs it asks for
        totals['gpuh'] += (8/60.0 if a.test else wall)*ng
        totals['useful'] += sum(float(r['est_wall_h'])*int(r['gpus'])*EFF.get(int(r['gpus']), 0.6)/EFF.get(g, 0.6) for r in rs)/a.speed/a.node_gpus
        totals['disk'] += sum(float(r['est_disk_gb']) for r in rs)

    print(f"{len(sel)} runs in {len(plan)} jobs on {part or '<partition>'} ({a.node_gpus} GPUs per node, {gmem:.0f} GB cards), "
          f"{'8-minute test in test/' if a.test else 'real runs'}")
    for k, (rs, g, wall, limit, nseg, cmd, env) in enumerate(plan, 1):
        print(f"\njob {k}: {g} GPU{'s' if g > 1 else ''} per run, limit {limit}" + (f', about {nseg} segment{"s" if nseg > 1 else ""}' if nseg else '')
              + ('' if a.test else f", longest run about {wall:.1f} h at speed {a.speed:g} = {wall:.1f} node-hours or {wall*(len(rs)*g if a.exact_gpus else a.node_gpus):.1f} GPU-hours"))
        for r in rs:
            print(f"    {r['id']:20s} stage {r['stage']:3s} theta {r['theta_deg']:>5s}  grid {r['nx']}x{r['ny']}x{r['nz']}"
                  f"  est {float(r['est_wall_h']):5.1f} h  {float(r['est_mem_gb']):4.1f} GB")
    if a.test:
        print(f"\ntest: {totals['nodeh']:.2f} node-hours at most (8 minutes per job)")
    else:
        useful_gpuh = totals['useful']*a.node_gpus
        idle = 1 - useful_gpuh/totals['gpuh'] if totals['gpuh'] else 0.0
        print(f"\nestimate at speed {a.speed:g} (the A100 figures of the manifest = 1), {totals['disk']/1000:.2f} TB of output in {base}/:")
        print(f"  billed per node:  {totals['nodeh']:.0f} node-hours for this plan (a job is one node, however many GPUs it asks for)")
        print(f"  billed per GPU:   {totals['gpuh']:.0f} GPU-hours for this plan (the GPUs each job holds; {100*idle:.0f} % of them idle while a node waits for its longest run); "
              f"{useful_gpuh:.0f} GPU-hours with one job per run (--pack 1 --exact-gpus)")
        print("  (the documentation says projects are granted GPU hours; check what your project is billed: README, 'Billing')")
    if a.dry_run:
        for rs, g, wall, limit, nseg, cmd, env in plan:
            print('\nRUNS="%s" RUN_GPUS=%d %s' % (env['RUNS'], g, ' '.join(cmd)))
        return
    if not a.account or not part:
        sys.exit('set HST_ACCOUNT and HST_PARTITION (or --account and --partition)')
    os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    if a.test:
        for rs, *_ in plan:
            for r in rs:
                shutil.rmtree(os.path.join(HERE, 'test', r['id']), ignore_errors=True)
                os.makedirs(os.path.join(HERE, 'test', r['id']))
                shutil.copy(os.path.join(HERE, 'runs', r['id'], 'hst.in'), os.path.join(HERE, 'test', r['id']))
    for rs, g, wall, limit, nseg, cmd, env in plan:
        out = subprocess.run(cmd, env=env, cwd=HERE, capture_output=True, text=True)
        if out.returncode:
            sys.exit(f'sbatch failed: {out.stderr.strip()}')
        jid = out.stdout.strip().split(';')[0]
        print(f"submitted job {jid}: {' '.join(r['id'] for r in rs)}")
        if real:
            with open(log, 'a') as f:
                for r in rs:
                    f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}\t{jid}\t{r['id']}\t{r['stage']}\t{part}\n")
    print('\nsqueue -u $USER;  ./status.py' + (' --base test' if a.test else ''))


if __name__ == '__main__':
    main()
