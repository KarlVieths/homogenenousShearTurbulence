#!/bin/bash
# HoreKA CPU job: the CPL post-processing (jobs/cpl_postprocess.sh) of a run
# directory.  postpro.cpl holds one whole velocity field per rank plus its
# nine derivatives and the spectra (about 50 GB per rank at 1536 x 1024 x 512),
# so RANKS is bounded by the node's memory, not its cores.
#   sbatch --export=ALL,RUNDIR=/hkfs/work/workspace/scratch/xt8786-hst/re20000,NFMIN=6 ~/hst/jobs/horeka_postprocess.slurm
# RANKS (4), NFMIN (1), NFMAX (all fields), DN (1), HST_ROOT (~/hst) and
# HST_MAIN (~/hst-main, an rsync copy of the CPL code) are overridable.
#SBATCH --job-name=hst-post
#SBATCH --account=hk-project-exasim
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=19
#SBATCH --mem=230gb
#SBATCH --time=06:00:00
#SBATCH --output=hst-post-%j.out

HST_ROOT=${HST_ROOT:-$HOME/hst}
export HST_MAIN=${HST_MAIN:-$HOME/hst-main}
RANKS=${RANKS:-4}
[ -n "$RUNDIR" ] || { echo "RUNDIR must be set (--export=ALL,RUNDIR=...)"; exit 1; }
module purge; module load compiler/gnu/13; module load mpi/openmpi/5.0
export PATH=$HOME/.local/bin:$PATH
nf=$(ls $RUNDIR/fields/field*.fld | wc -l)
echo "job $SLURM_JOB_ID on $(hostname): $RUNDIR, fields ${NFMIN:-1}..${NFMAX:-$nf} by ${DN:-1} on $RANKS ranks"
time $HST_ROOT/jobs/cpl_postprocess.sh $RUNDIR $RANKS ${NFMIN:-1} ${NFMAX:-$nf} ${DN:-1}
