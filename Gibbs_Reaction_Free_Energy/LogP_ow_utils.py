import concurrent.futures
import os
import random
import shutil
import string
import subprocess
from datetime import datetime

import numpy as np
from rdkit import Chem


def write_xyz_file4xtb(args): 
    fragment, solvent, filename, destination = args
    number_of_atoms = fragment.GetNumAtoms()
    symbols = [a.GetSymbol() for a in fragment.GetAtoms()]
    conformers = fragment.GetConformers()
    solvent_dir = os.path.join(destination, solvent)
    os.makedirs(solvent_dir)
    file_name = f"{filename}.xyz"
    file_path = os.path.join(solvent_dir, file_name)
    for i, conf in enumerate(conformers):
        with open(file_path, "w") as _file:
            _file.write(str(number_of_atoms) + "\n")
            _file.write(f"{Chem.MolToSmiles(fragment)}\n")
            for atom, symbol in enumerate(symbols):
                p = conf.GetAtomPosition(atom)
                line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
                _file.write(line)
    return file_path

def read_G_mol(output, err):
    if not "normal termination" in err:
        raise Warning(err)
    lines = output.splitlines()
    G_mol = None
    for l in lines:
        if "TOTAL FREE ENERGY" in l:
            G_mol = float(l.split()[4])
    return G_mol

def LogP_ow(args):
    linker_conformer, solvent, LogP_dir, chrg, numThreads, name = args

    xtb_input_args = (linker_conformer,
                       solvent,
                       "mol4LogP",
                       LogP_dir)
    path_xtb_input = write_xyz_file4xtb(xtb_input_args)
    cwd = os.path.dirname(path_xtb_input)
    xyz_file = os.path.basename(path_xtb_input)

    xtb_modload = "module load xtb/6.5.1;"
    xtb_options = f"{xyz_file} --chrg {chrg} --alpb {solvent} --ohess tight --uhf 0 "
    cmd_xtb = f"{xtb_modload} xtb {xtb_options} | tee xtb.out"
    os.environ["OMP_NUM_THREADS"] = f"{numThreads},1"
    os.environ["MKL_NUM_THREADS"] = f"{numThreads}"
    os.environ["OMP_STACKSIZE"] = "200G"
    print(f"calculating {name} G_{solvent} for LogP on {numThreads} core(s) starting at {datetime.now()}")
    popen = subprocess.Popen(
        cmd_xtb,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
        cwd=cwd,
    )
    popen.communicate()
    output, err = popen.communicate()
    G_mol = read_G_mol(output, err)
    if G_mol == None:
        print(f"Error: Final Gibss free energy not found.")
        return None 
    else:
        return G_mol
