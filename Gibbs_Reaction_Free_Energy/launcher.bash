#!/bin/bash

function STDV {
	array=("$@")
	sum=0
	sum2=0
	for i in "${array[@]}"; do
		sum=`echo "$sum + $i" | bc -l`
		sum2=`echo "$sum2 + $i^2" | bc -l`
	done
	N=`echo ${#array[@]}`
	stdv=`echo "sqrt($sum2/$N - ($sum/$N)^2)" |bc -l`
	avg=`echo "$sum/$N" | bc -l`
	echo "$avg $stdv"
}

#Gsolv=()
#qcg_E=()

nsolv=0
trials=1
mdtime=80 #trials and mtime will change with the same index
cmplx_file=$1
nloop=0 # I should add +1 to this index to change the number of trials
root_dir=$(pwd)

for i in {1..3}; do

	mkdir nsolv-"$nsolv"_"$i"

	cp CENSO_jobfile.bash CREST_jobfile.bash $cmplx_file H2O.xyz nsolv-"$nsolv"_"$i"
	cd nsolv-"$nsolv"_"$i"


	##############################
	## CREST cluster generation ##
	##############################

	crest_jobid=$(sbatch --parsable --job-name=CRST_Shnk-cmplx_"$nsolv"-"$i" --exclude=node3 CREST_jobfile.bash $nsolv $mdtime $cmplx_file)

	#######################
	## CENSO calculation ##
	#######################

       	sbatch --job-name=CNS_Shnk-cmplx_"$nsolv"-"$i" --exclude=node4,node5,node7 --dependency=afterok:$crest_jobid CENSO_jobfile.bash 

	cd $root_dir
done
