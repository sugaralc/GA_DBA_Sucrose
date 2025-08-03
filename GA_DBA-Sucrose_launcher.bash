#!/bin/bash
#SBATCH --job-name=GA_tun_0.5Mut1.5Pres
#SBATCH --output=logfile.GA_Sucrose-DBA.%A.out
#SBATCH --error=errorfile.GA_Sucrose-DBA.%A.err
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=5G
##SBATCH --exclude=hnode1,hnode4
##SBATCH --array=[433-1437]%50

source /users/glara/anaconda3/bin/activate rdkit-env
scratch_dir=/scratch/glara/GA_DBA-Sucrose_job_$SLURM_JOB_ID


mkdir -p $scratch_dir
idx=$SLURM_JOB_ID

cp -r * $scratch_dir
cd $scratch_dir

python3 GA_Sucrose-DBA_tweezer.py
#mkdir $SLURM_SUBMIT_DIR/results_GA_DBA-Sucrose_$SLURM_JOB_ID
#cp -r * $SLURM_SUBMIT_DIR/results_GA_DBA-Sucrose_$SLURM_JOB_ID
#rm -r $scratch_dir

