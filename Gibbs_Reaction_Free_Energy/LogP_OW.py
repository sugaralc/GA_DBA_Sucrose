import concurrent.futures
import os
import random
import shutil
import string
import subprocess
from datetime import datetime

import numpy as np
os.environ["SCRATCH"] = f"/scratch/glara"

def write_orca_inputs(args):
    path_censobest = args[0]
    destination = args[1]
    solvent = args[2]
    numThreads = args[3]
    charge = args[4]
    spin = args[5]
    #Add here the agrs way to pass the arguments from one side to the other.   
    #LogP_path = os.path.join(destination, "LogP")
    #os.makedirs(LogP_path)
    if solvent == "1-octanol":
        path_orca_workdir = os.path.join(destination, "octanol")
        os.makedirs(path_orca_workdir) 
        file_name = "G_octanol.inp"
        orca_input_path = os.path.join(path_orca_workdir,file_name)
    elif solvent == "water":
        path_orca_workdir = os.path.join(destination, "water")
        os.makedirs(path_orca_workdir)
        file_name = "G_water.inp"
        orca_input_path = os.path.join(path_orca_workdir,file_name)

    with open(orca_input_path, "w") as _file:
            _file.write(f"!Opt r2scan-3c VeryTightSCF NumFreq\n")
            _file.write(f'%base "Gibss_free_energy"\n')
            _file.write(f'%pal nproc {numThreads}\n')
            _file.write(f'end\n')
            _file.write(f'%maxcore 10000\n')
            _file.write(f'%cpcm\n')
            _file.write(f'smd true\n')
            _file.write(f'SMDsolvent "{solvent}"\n')
            _file.write(f'end\n')
            #_file.write(f'%geom\n')
            #_file.write(f'maxiter 1000\n')
            #_file.write(f'end\n')
            _file.write(f'\n')
            _file.write(f'* xyzfile {charge} {spin} censo_best.xyz\n')
            _file.write(f'\n')
    
    #conf_ensemble = os.path.basename(conf_ensemble_path)
    src = f"{path_censobest}/coord.enso_best"
    dst = f'{path_orca_workdir}/censo_best.tmol'
    shutil.copy(src, dst)
    return orca_input_path

def read_G_mol(output, err):
    if not "****ORCA TERMINATED NORMALLY****" in output:
        raise Warning(err)
    lines = output.splitlines()
    G_mol = None
    for l in lines:
        if "Final Gibbs free energy" in l:
            G_mol = float(l.split()[5])
    return G_mol

#def LogP_ow(
#    censo_dir=None,
#    solvent="1-octanol",
#    LogP_dir=None,
#    charge=-2,
#    spin=1,
#    numThreads=1
#    ): #Here added the input arguments including the most estable conformer from censo

def LogP_ow(args):
    censo_dir, solvent, LogP_dir, charge, numThreads = args
    spin = 1 
    orca_input_args = (censo_dir,LogP_dir,solvent,numThreads,charge,spin)
    path_orca_input = write_orca_inputs(orca_input_args)
    cwd = os.path.dirname(path_orca_input)
    orca_input = os.path.basename(path_orca_input)
    orca_modload = "module load ORCA/5.0.4;"
    tmol2xyz = "obabel censo_best.tmol -O censo_best.xyz;"
    cmd_orca = f"{tmol2xyz} {orca_modload} $ORCA_BIN/orca {orca_input} | tee orca.out"
    print(f"calculating LogP for the free ligand on {numThreads} core(s) starting at {datetime.now()}")
    popen = subprocess.Popen(
        cmd_orca,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
        cwd=cwd,
    )

    if popen.wait() == 0:
        output, err = popen.communicate()
        G_mol = read_G_mol(output, err)
        if G_mol == None:
            print(f"Error: Final Gibss free energy not found.")
            return None 
        else:
            return G_mol

if __name__ == "__main__":
    write_orca_inputs(xyz_file="censo_best.xyz")    