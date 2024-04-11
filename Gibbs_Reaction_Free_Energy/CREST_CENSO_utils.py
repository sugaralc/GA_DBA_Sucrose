import concurrent.futures
import os
import random
import shutil
import string
import subprocess
from datetime import datetime

import numpy as np
from rdkit import Chem
# Here create a function called molecular_free_energy, 
#def molecular_free_energy() This function should return the G(X) for the complex and
os.environ["SCRATCH"] = f"/scratch/glara"

def write_xyz_file4crest(fragment, name, destination="."):#Here modify to have the correct directory name 
    number_of_atoms = fragment.GetNumAtoms()
    symbols = [a.GetSymbol() for a in fragment.GetAtoms()]
    conformers = fragment.GetConformers()
    file_paths = []
    for i, conf in enumerate(conformers):
        crest_path = os.path.join(destination, f"crest_job")
        os.makedirs(crest_path)
        file_name = f"{name}.xyz"
        file_path = os.path.join(crest_path, file_name)
        with open(file_path, "w") as _file:
            _file.write(str(number_of_atoms) + "\n")
            _file.write(f"{Chem.MolToSmiles(fragment)}\n")
            for atom, symbol in enumerate(symbols):
                p = conf.GetAtomPosition(atom)
                line = " ".join((symbol, str(p.x), str(p.y), str(p.z), "\n"))
                _file.write(line)
        file_paths.append(file_path)
    return file_name

def write_input_toml4crest(CREST_OPTIONS, destination="."):
    crest_path = os.path.join(destination, f"crest_job")
    file_name = f"input.toml"
    file_path = os.path.join(crest_path, file_name)
    with open(file_path, "w") as _file:
        #_file.write(chr(35)This is a CREST input file)\n")
        _file.write('#This is a CREST input file\n')
        _file.write(f'input="{CREST_OPTIONS.get("input")}"\n')
        _file.write(f'threads={CREST_OPTIONS.get("threads")}\n')
        _file.write(f'\n')
        _file.write(f'#Metadynamics configuration\n')
        _file.write(f'[calculation]\n')
        _file.write(f'[[calculation.level]]\n')
        _file.write(f'binary={CREST_OPTIONS.get("binary")}\n')
        _file.write(f'method={CREST_OPTIONS.get("dyn_method")}\n')
        _file.write(f'weight=1.0\n')
        _file.write(f'charge={CREST_OPTIONS.get("charge")}\n')
        _file.write(f'uhf={CREST_OPTIONS.get("uhf")}\n')
        _file.write(f'gbsa={CREST_OPTIONS.get("gbsa")}\n')
        _file.write(f'\n')
        _file.write(f'#Optimization configuration\n')
        _file.write(f'[[calculation.level]]\n')
        _file.write(f'method={CREST_OPTIONS.get("opt_method")}\n')
        _file.write(f'charge={CREST_OPTIONS.get("charge")}\n')
        _file.write(f'uhf={CREST_OPTIONS.get("uhf")}\n')
        _file.write(f'gbsa={CREST_OPTIONS.get("gbsa")}\n')        

def run_crest(args):
    (xyz_files, crest_cmd, numThreads, crest_version) = args
    print(f"running {xyz_files} on {numThreads} core(s) starting at {datetime.now()}")
    cwd = os.path.dirname(xyz_files)
    xyz_file = os.path.basename(xyz_files)
    #user_modload = "module load user_modfiles;"
    crest_modload = "module load crest/2.12;"
    xtb_modload = "module load xtb/6.6.0;"
    slurm_modules = " ".join([crest_modload, xtb_modload])
    cmd = f"{slurm_modules} crest-2.12 {xyz_file} {crest_cmd} | tee crest.out" #check here
    os.environ["OMP_NUM_THREADS"] = f"{numThreads},1"
    os.environ["MKL_NUM_THREADS"] = f"{numThreads}"
    os.environ["OMP_STACKSIZE"] = "200G"
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
        cwd=cwd,
    )
    output, err = popen.communicate()
    energy = read_energy(output, err)
    return energy

def read_energy(output, err):
    if not "normal termination" in err:
        raise Warning(err)
    lines = output.splitlines()
    energy = None
    structure_block = False
    atoms = []
    coords = []
    for l in lines:
        if "final structure" in l:
            structure_block = True
        elif structure_block:
            s = l.split()
            if len(s) == 4:
                atoms.append(s[0])
                coords.append(list(map(float, s[1:])))
            elif len(s) == 0:
                structure_block = False
        elif "TOTAL ENERGY" in l:
            energy = float(l.split()[3])
    return energy, {"atoms": atoms, "coords": coords}

def molecular_free_energy(
    mol,
    solvent="h2o",
    #alpb=None,
    mdtime="x1",
    input=None,
    name=None,
    cleanup=False,
    numThreads=38,
    crest_version="2.12"
):
    # check mol input
    assert isinstance(mol, Chem.rdchem.Mol)
    if mol.GetNumAtoms(onlyExplicit=True) < mol.GetNumAtoms(onlyExplicit=False):
        raise Exception("Implicit Hydrogens")
    conformers = mol.GetConformers()
    n_confs = len(conformers)
    if not conformers:
        raise Exception("Mol is not embedded")
    elif not conformers[-1].Is3D():
        raise Exception("Conformer is not 3D")

    if not name:
        name = "tmp_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=4))

    # set SCRATCH if environmental variable
    try:
        scr_dir = os.environ["SCRATCH"]
    except:
        scr_dir = os.getcwd()
    print(f"SCRATCH DIR = {scr_dir}")

    charge = Chem.GetFormalCharge(mol)
    xyz_files = write_xyz_file4crest(mol, "crestmol", destination=os.path.join(scr_dir, name))

    # xtb options
    cmd=""
    if crest_version == "3.0":
        CREST_OPTIONS = {
            "binary": str('"xtb"'),
            "threads": numThreads,
            "charge": charge,
            "gbsa": solvent,
            "dyn_method": str('"gfnff"'),
            "opt_method": str('"gfn2"'),
            #"mdtime": mdtime,
            "uhf": 0,
            "input": xyz_files,
            }
        write_input_toml4crest(CREST_OPTIONS, destination=os.path.join(scr_dir, name))
    else:
        CREST_OPTIONS = {
            "threads": numThreads,
            "charge": charge,
            "gbsa": solvent,
            "uhf": "0",
            "mdtime": mdtime,
        }
        for key, value in CREST_OPTIONS.items():
            if value:
                cmd += f" --{key} {value}"

    workers = np.min([numThreads, n_confs])
    cpus_per_worker = numThreads // workers
    #args = [(xyz_file, cmd, cpus_per_worker) for xyz_file in xyz_files]
    args = (xyz_files, cmd, cpus_per_worker, crest_version)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(run_crest, args) #Continue here to determine how to pass the input file
                                                #See CREST 3.0 to see the new keyword for input

    CENSO_OPTIONS = {
        "input": "crest_conformers.xyz",
        "solvent": solvent,
        "charge": charge,
        "balance": "on",
        "part0": "on",
        "thresholdpart0": 10.0,
        "part1": "on",
        "thresholdpart1": 6.0,
        "part2": "off",
        "thresholdpart2": 3.0,
    }

    energies = []
    geometries = []
    for e, g in results:
        energies.append(e)
        geometries.append(g)

    minidx = np.argmin(energies)

    # Clean up
    if cleanup:
        shutil.rmtree(name)