#!/bin/bash
#SBATCH --job-name=CREST-screen
#SBATCH --output=logfile.crest-Complex.%A.out
#SBATCH --error=errorfile.crest-Complex.%A.err
#SBATCH --partition=FULL
#SBATCH --nodes=1
#SBATCH --tasks-per-node=40
#SBATCH --mem=5G
#SBATCH --exclude=node3
##SBATCH --array=[433-1437]%50

str=$1
mdt=$2
cmplx_file=$3

module load xtb/6.6.0

export OMP_NUM_THREADS=38,1
export MKL_NUM_THREADS=38
ulimit -s unlimited
export OMP_STACKSIZE=35G
export OMP_MAX_ACTIVE_LEVELS=1

exe_dir=$(pwd)

scratch_dir=/scratch/glara/crestjob_$SLURM_JOB_ID

mkdir -p $scratch_dir
idx=$SLURM_JOB_ID

cp $cmplx_file H2O.xyz $scratch_dir
cd $scratch_dir

if [[ $str -eq 0 ]]
then
	crest $cmplx_file --chrg -2 --T 38 --mdtime $mdt --g h2o > CREST_cluster_generation_nsolv-"$str".out
else
	crest $cmplx_file --qcg H2O.xyz --chrg -2 --nsolv $str --T 38 --ensemble --mdtime $mdt --alpb water --wscal 1.0 --nofix --nclus 10 -gnf2 > CREST_cluster_generation_nsolv-"$str".out
fi

mkdir $SLURM_SUBMIT_DIR/results_crest_nsolv-"$str"_job_"$SLURM_JOB_ID"
#cp ensemble/full_ensemble.xyz $SLURM_SUBMIT_DIR
if [[ $str -eq 0 ]]
then
	cp crest_conformers.xyz $SLURM_SUBMIT_DIR
else
	cp ensemble/full_ensemble.xyz $SLURM_SUBMIT_DIR
fi

cp -r * $SLURM_SUBMIT_DIR/results_crest_nsolv-"$str"_job_"$SLURM_JOB_ID"
#rm -r $scratch_dir
