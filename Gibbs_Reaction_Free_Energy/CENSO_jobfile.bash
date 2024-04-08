#!/bin/bash
#SBATCH --job-name=CENSO-screen
#SBATCH --output=logfile.censo-Complex.%A.out
#SBATCH --error=errorfile.censo-Complex.%A.err
#SBATCH --partition=FULL
#SBATCH --nodes=1
#SBATCH --tasks-per-node=40
#SBATCH --mem=300G
##SBATCH --exclude=hnode1,hnode4
##SBATCH --array=[433-1437]%50

infile=crest_conformers.xyz

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

module load xtb/6.5.1
module load ORCA/5.0.4

export OMP_NUM_THREADS=38,1
export MKL_NUM_THREADS=38
ulimit -s unlimited
export OMP_STACKSIZE=35G
export OMP_MAX_ACTIVE_LEVELS=1

exe_dir=$(pwd)

scratch_dir=/scratch/glara/censo_job_$SLURM_JOB_ID

mkdir -p $scratch_dir
idx=$SLURM_JOB_ID

cp $infile $scratch_dir
cd $scratch_dir

export PYTHONUNBUFFERED=1
censo_hf-3c_2 -inp $infile -solvent h2o -chrg -2 -P 38 -O 1 -balance on -thrpart0 10.0 -thrpart1 6.0 -thrpart2 3.0 > CENSO_sorting.out
mkdir $SLURM_SUBMIT_DIR/results_censo_job_$SLURM_JOB_ID
cp * $SLURM_SUBMIT_DIR/results_censo_job_$SLURM_JOB_ID
#rm -r $scratch_dir

