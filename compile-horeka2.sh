#!/usr/bin/env bash
# Build the executables for HoreKA2, each ON the kind of node it will run on.
#
#   ./compile-horeka2.sh                    h100, h200, b200 and cpu: one build job per GPU type, in parallel, wait for them
#   ./compile-horeka2.sh h100 b200          only these
#   ./compile-horeka2.sh --dev b200         build on the development partition (1 h limit, one job at a time, so sequential)
#   ./compile-horeka2.sh --no-wait h200     submit and return (the jobs are listed; results in build-logs/)
#   ./compile-horeka2.sh --status           what is built, from which commit, and whether the source changed since
#   ./compile-horeka2.sh --local h100       build HERE, on this node, as the old script did (the architecture must match: see below)
#
# Why jobs: the B200 nodes (HoreKa Jade) have ARM Grace CPUs while the login nodes are x86-64, so a B200 binary built on a
# login node does not run on its nodes; and a build on the node itself is also tested there.  Each job
#   1. checks the node: the GPU name (H100, H200 or B200) and the CPU architecture (x86_64; aarch64 for the B200);
#   2. sources env/horeka2.sh for the device and builds into build-gpu-<device>.new  (make GPU=1 NCCL=1);
#   3. checks that the binary is for this architecture, and runs the repository's small deck (hst.in) for 20 steps on one GPU;
#   4. writes build-gpu-<device>/BUILD_INFO (commit, dirty or not, node, architecture, GPU, driver, compiler, modules, speed of the test)
#      and only then replaces build-gpu-<device> (the previous build stays as build-gpu-<device>.old).
# A failed build leaves the previous binary alone.  The job logs are build-logs/hst-build-<device>-<jobid>.out, the verdict
# build-logs/result-<device>.  cpu is built on the node you run the script on (x86-64 login node), as before.
#
# Environment: HST_ACCOUNT (passed as --account if set), HST_BUILD_TIME (default 00:45:00), HST_PARTITION_<DEVICE> to override a
# partition (e.g. HST_PARTITION_B200=gpu-b200).  The H100 and H200 builds are separate directories although both are sm_90.
set -uo pipefail

ROOT=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
cd "$ROOT" || exit 1
LOGS=$ROOT/build-logs

# device -> (GPU name pattern, CPU architecture, GPU_ARCH, partition, development partition)
gpu_name()   { case $1 in h100) echo H100;; h200) echo H200;; b200) echo B200;; esac; }
gpu_cpu()    { case $1 in h100|h200) echo x86_64;; b200) echo aarch64;; esac; }
gpu_cc()     { case $1 in h100|h200) echo cc90;; b200) echo cc100;; esac; }
gpu_part()   { case $1 in h100) echo "${HST_PARTITION_H100:-gpu-h100}";; h200) echo "${HST_PARTITION_H200:-gpu-h200}";; b200) echo "${HST_PARTITION_B200:-gpu-b200}";; esac; }
gpu_dev()    { case $1 in h100) echo dev-gpu-h100;; b200) echo dev-gpu-b200;; h200) echo "";; esac; }   # H200 has no development partition

build_info() {   # BUILD_INFO text of the node we are on
  local t=$1 dirty=clean
  [ -n "$(git -C "$ROOT" status --porcelain -- src Makefile env 2>/dev/null)" ] && dirty="DIRTY (uncommitted changes in src/, Makefile or env/)"
  echo "device:    $t"
  echo "commit:    $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null) on $(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null), $dirty"
  echo "built:     $(date '+%Y-%m-%d %H:%M:%S') on $(hostname), $(uname -m)"
  echo "gpu:       $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -n1)"
  echo "GPU_ARCH:  ${GPU_ARCH:-?}"
  echo "compiler:  $(nvfortran --version 2>/dev/null | sed -n 2p)"
  echo "modules:   $(module -t list 2>&1 | tr '\n' ' ')"
}

# ---------------------------------------------------------------- on the compute node
build_on_node() {
  local t=$1 B=build-gpu-$1
  mkdir -p "$LOGS"; rm -f "$LOGS/result-$t"
  die() { echo "BUILD FAILED: $*"; echo "failed: $*" > "$LOGS/result-$t"; rm -rf "$ROOT/$B.new"; exit 1; }
  local want_gpu want_cpu want_cc
  want_gpu=$(gpu_name "$t"); want_cpu=$(gpu_cpu "$t"); want_cc=$(gpu_cc "$t")
  [ -n "$want_gpu" ] || die "unknown device '$t'"
  local have_gpu; have_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
  echo "node $(hostname), $(uname -m), GPU '$have_gpu', build target $t"
  case "$have_gpu" in *"$want_gpu"*) ;; *) die "this node's GPU is '${have_gpu:-none}', not $want_gpu (wrong partition?)";; esac
  [ "$(uname -m)" = "$want_cpu" ] || die "this node is $(uname -m), the $t binary needs $want_cpu"

  # shellcheck disable=SC1091
  source env/horeka2.sh gpu "$t" || die "env/horeka2.sh failed (module names on this node's software tree? try: module avail NVHPC)"
  [ "${GPU_ARCH:-}" = "$want_cc" ] || die "GPU_ARCH is '${GPU_ARCH:-}', expected $want_cc"
  echo "GPU_ARCH=$GPU_ARCH NVHPC_ROOT=${NVHPC_ROOT:-?}"; module list 2>&1 | tail -6

  rm -rf "$ROOT/$B.new"
  make -j"${SLURM_CPUS_PER_TASK:-8}" GPU=1 NCCL=1 BUILD="$B.new" > "$LOGS/make-$t.log" 2>&1 || { tail -25 "$LOGS/make-$t.log"; die "make failed (log: build-logs/make-$t.log)"; }
  tail -3 "$LOGS/make-$t.log"
  [ -x "$B.new/hst" ] || die "no $B.new/hst after make"
  if command -v file >/dev/null 2>&1; then
    local ft; ft=$(file -L -b "$B.new/hst")
    case "$want_cpu:$ft" in x86_64:*x86-64*|aarch64:*aarch64*) ;; *) die "binary is not $want_cpu: ${ft:0:80}";; esac
  fi

  # smoke test: the small deck of the repository, 20 steps, one rank on one GPU
  local sm=$LOGS/smoke-$t sps=
  rm -rf "$sm"; mkdir -p "$sm"
  sed -e 's/nstep *= *[0-9]*/nstep = 20/' -e 's/wall_max *= *[0-9.eE+-]*/wall_max = 0.0/' hst.in > "$sm/hst.in"
  ( cd "$sm" && CUDA_VISIBLE_DEVICES=0 timeout 600 mpirun --bind-to none -np 1 "$ROOT/$B.new/hst" hst.in > run.log 2>&1 )
  grep -q "end of run" "$sm/run.log" || { tail -15 "$sm/run.log"; die "the 20-step test run did not finish (build-logs/smoke-$t/run.log)"; }
  grep -q "ERROR" "$sm/run.log" && { grep "ERROR" "$sm/run.log" | head -5; die "the test run reports an error (build-logs/smoke-$t/run.log)"; }
  sps=$(grep -o '[0-9.]* s/step' "$sm/run.log" | tail -1)
  echo "smoke test: ok ($sps)"

  { build_info "$t"; echo "smoke:     20 steps of hst.in, ${sps:-?}"; echo "status:    ok"; } > "$B.new/BUILD_INFO"
  rm -rf "$B.old"; [ -d "$B" ] && mv "$B" "$B.old"
  mv "$B.new" "$B"
  echo "ok $(hostname) $(uname -m) ${sps}" > "$LOGS/result-$t"
  echo "installed $ROOT/$B/hst"; cat "$B/BUILD_INFO"
}

# ---------------------------------------------------------------- the CPU build, here
build_cpu_here() {
  ( set -e; source env/horeka2.sh cpu; make -j8 BUILD=build-cpu )
}

# ---------------------------------------------------------------- status
show_status() {
  for t in h100 h200 b200 cpu; do
    local d=build-gpu-$t; [ $t = cpu ] && d=build-cpu
    if [ ! -x "$d/hst" ]; then printf '%-5s not built\n' $t; continue; fi
    if [ -f "$d/BUILD_INFO" ]; then
      local c; c=$(sed -n 's/^commit: *\([0-9a-f]*\) .*/\1/p' "$d/BUILD_INFO")
      local note=
      if [ -n "$c" ] && git cat-file -e "$c" 2>/dev/null; then
        git diff --quiet "$c" HEAD -- src Makefile env 2>/dev/null || note="  STALE: src/Makefile/env changed since $c"
      fi
      grep -q DIRTY "$d/BUILD_INFO" && note="$note  (built from uncommitted changes)"
      printf '%-5s %s | %s%s\n' $t "$(sed -n 's/^commit: *//p' "$d/BUILD_INFO" | cut -c1-40)" "$(sed -n 's/^built: *//p' "$d/BUILD_INFO")" "$note"
    else
      printf '%-5s %s (no BUILD_INFO: built by hand or by an older script; %s)\n' $t "$d/hst" "$(file -L -b "$d/hst" 2>/dev/null | cut -c1-40)"
    fi
  done
}

# ---------------------------------------------------------------- main
if [ "${1:-}" = --in-job ]; then build_on_node "$2"; exit $?; fi

DEV=0; WAIT=1; LOCAL=0; STATUS=0; targets=()
while [ $# -gt 0 ]; do
  case $1 in
    --dev) DEV=1;; --no-wait) WAIT=0;; --local) LOCAL=1;; --status) STATUS=1;;
    -h|--help) sed -n '2,24p' "$0"; exit 0;;
    h100|h200|b200|cpu) targets+=("$1");;
    *) echo "unknown argument '$1' (targets: h100 h200 b200 cpu)"; exit 2;;
  esac; shift
done
[ $STATUS = 1 ] && { show_status; exit 0; }
[ ${#targets[@]} -eq 0 ] && targets=(h100 h200 b200 cpu)

rc=0; cpu_done=0
do_cpu() {
  [ $cpu_done = 1 ] && return
  for t in "${targets[@]}"; do
    if [ "$t" = cpu ]; then echo ">>> cpu built HERE ($(hostname))"; build_cpu_here || { echo "cpu: build failed"; rc=1; }; fi
  done
  cpu_done=1
}
gpus=(); for t in "${targets[@]}"; do [ "$t" = cpu ] || gpus+=("$t"); done

if [ $LOCAL = 1 ]; then
  for t in "${gpus[@]}"; do
    if [ "$(uname -m)" != "$(gpu_cpu "$t")" ]; then echo "--local $t: this node is $(uname -m), the $t binary needs $(gpu_cpu "$t")"; rc=1; continue; fi
    echo ">>> $t built HERE ($(hostname), no GPU test unless this node has the GPU)"
    ( set -e; source env/horeka2.sh gpu "$t"; B=build-gpu-$t; rm -rf "$B.new"; make -j8 GPU=1 NCCL=1 BUILD="$B.new"
      { build_info "$t"; echo "smoke:     none (built with --local)"; echo "status:    built, untested"; } > "$B.new/BUILD_INFO"
      rm -rf "$B.old"; [ -d "$B" ] && mv "$B" "$B.old"; mv "$B.new" "$B" ) || { echo "$t: build failed"; rc=1; }
  done
else
  mkdir -p "$LOGS"
  declare -A jid
  submit() {   # submit the build job of one device, print its job id
    local t=$1 part; part=$(gpu_part "$t"); [ $DEV = 1 ] && [ -n "$(gpu_dev "$t")" ] && part=$(gpu_dev "$t")
    rm -f "$LOGS/result-$t"
    local opts=(--parsable --job-name="hst-build-$t" --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=16
                --time="${HST_BUILD_TIME:-00:45:00}" --partition="$part" --chdir="$ROOT" --output="$LOGS/%x-%j.out" --export=ALL)
    [ -n "${HST_ACCOUNT:-}" ] && opts+=(--account="$HST_ACCOUNT")
    sbatch "${opts[@]}" --wrap="'$ROOT/compile-horeka2.sh' --in-job $t" | cut -d';' -f1
  }
  wait_for() {   # until the jobs are gone from the queue (no polling faster than 30 s: HoreKA2 documentation)
    local ids=$1
    while [ -n "$(squeue -h -j "$ids" -o %i 2>/dev/null)" ]; do sleep 30; done
  }
  report() {
    local t=$1
    if [ -f "$LOGS/result-$t" ] && grep -q '^ok' "$LOGS/result-$t"; then
      echo ">>> $t: $(cat "$LOGS/result-$t")"; sed -n 's/^commit: *//p;s/^built: *//p' "build-gpu-$t/BUILD_INFO" | sed 's/^/        /'
    else
      echo ">>> $t: FAILED - $(cat "$LOGS/result-$t" 2>/dev/null || echo "no result file: the job did not run to the end, see sacct -j ${jid[$t]:-?} and build-logs/")"; rc=1
    fi
  }
  if [ $DEV = 1 ]; then          # one job at a time on the development partitions
    for t in "${gpus[@]}"; do
      id=$(submit "$t") || { echo "$t: sbatch failed"; rc=1; continue; }
      jid[$t]=$id; echo "$t: job $id on $(gpu_part "$t")$([ -n "$(gpu_dev "$t")" ] && echo ' (dev)')"
      [ $WAIT = 1 ] && { wait_for "$id"; report "$t"; }
    done
  else
    for t in "${gpus[@]}"; do
      id=$(submit "$t") || { echo "$t: sbatch failed"; rc=1; continue; }
      jid[$t]=$id; echo "$t: job $id on $(gpu_part "$t")"
    done
    do_cpu          # while the GPU jobs queue and build
    if [ $WAIT = 1 ] && [ ${#jid[@]} -gt 0 ]; then
      echo "waiting for the build jobs (squeue -u \$USER; logs in build-logs/) ..."
      wait_for "$(IFS=,; echo "${jid[*]}")"
      for t in "${gpus[@]}"; do [ -n "${jid[$t]:-}" ] && report "$t"; done
    fi
  fi
  [ $WAIT = 0 ] && echo "submitted; results in build-logs/result-<device> and build-gpu-<device>/BUILD_INFO, summary with ./compile-horeka2.sh --status"
fi

do_cpu          # (a no-op if it was built above)
exit $rc
