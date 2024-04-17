import concurrent.futures
import os
import random
import shutil
import string
import subprocess
from datetime import datetime

import numpy as np
from rdkit import Chem
#from LogP_OW import LogP_ow
from utils import hartree2kcalmol
R_gas = 1.98720425864083/1000
os.environ["SCRATCH"] = f"/scratch/glara"

def write_xyz_file4crest(fragment, name, destination="."): 
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
    return file_paths

def write_input_toml4crest(CREST_OPTIONS, destination="."):
    crest_path = os.path.join(destination, f"crest_job")
    file_name = f"input.toml"
    file_path = os.path.join(crest_path, file_name)
    with open(file_path, "w") as _file:
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
    xyz_files, crest_cmd, numThreads, crest_version = args
    print(f"running {xyz_files} on {numThreads} core(s) starting at {datetime.now()}")
    cwd = os.path.dirname(xyz_files)
    xyz_file = os.path.basename(xyz_files)
    #user_modload = "module load user_modfiles;"
    crest_modload = f"module load CREST/{crest_version};"
    xtb_modload = "module load xtb/6.6.0;"
    slurm_modules = " ".join([crest_modload, xtb_modload])
    cmd = f"{slurm_modules} crest-{crest_version} {xyz_file} {crest_cmd} | tee crest.out" #check here
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
    popen.communicate() 
    #if popen.wait() == 0:
    conf_ensemble_path = f'{cwd}/crest_conformers.xyz'
    try:
        f = open(conf_ensemble_path,'r')
        print(f'{conf_ensemble_path} found.')
        output, err = popen.communicate()
        S_conf = read_S_conf(output, err)
        if S_conf == None:
            print(f"Error: S_conf not found.")
            return None 
        else:
            return conf_ensemble_path, S_conf
    except FileNotFoundError:
        print(f"Error: {conf_ensemble_path} not found.")
        return None

def read_S_conf(output, err):
    if not "CREST terminated normally" in output:
        raise Warning(err)
    lines = output.splitlines()
    S_conf = None
    for l in lines:
        if "ensemble entropy" in l:
            S_conf = float(l.split()[8])
    return S_conf

def run_censo(args):
    conf_ensemble_path, censo_cmd, numThreads, destination = args
    cwd = os.path.join(destination, f"censo_job")
    os.makedirs(cwd)
    conf_ensemble = os.path.basename(conf_ensemble_path)
    src = f"{conf_ensemble_path}"
    dst = f"{cwd}/{conf_ensemble}"
    shutil.copy(src, dst)
    print(f"censo sorting of {conf_ensemble} on {numThreads} core(s) starting at {datetime.now()}")
    user_modload = "module load user_modfiles;"
    crest_modload = "module load censo/1.2.0_HF-3c_glara;"
    xtb_modload = "module load xtb/6.5.1;"
    orca_modload = "module load ORCA/5.0.4;"
    slurm_modules = " ".join([user_modload, crest_modload, orca_modload, xtb_modload])
    cmd = f"{slurm_modules} censo {censo_cmd} | tee censo.out" 
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
    popen.communicate()
    #if popen.wait() == 0: 
    output, err = popen.communicate()
    free_energy = read_free_energy(output, err)
    #I should take the file censo.best
    return free_energy, cwd

def read_free_energy(output, err):
    if not "CENSO all done!" in output:
        raise Warning(err)
    lines = output.splitlines()
    energy = None
    for l in lines:
        if "<<==part1==" in l:
            energy = float(l.split()[4])
    return energy

def write_orca_inputs(args):
    path_censobest = args[0]
    destination = args[1]
    solvent = args[2]
    numThreads = args[3]
    charge = args[4]
    spin = args[5]
    
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
    print(f"calculating free ligand's G_{solvent} for LogP on {numThreads} core(s) starting at {datetime.now()}")
    popen = subprocess.Popen(
        cmd_orca,
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

def molecular_free_energy(
    mol,
    solvent="h2o",
    #alpb=None,
    mdtime="x1",
    input=None,
    name=None,
    calc_LogP_OW=False,
    cleanup=False,
    numThreads=1,
    crest_version='2.12'
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

    # crest options
    cmd_crest = ""
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
            "T": numThreads,
            "chrg": charge,
            "g": solvent,
            "uhf": "0",
            "mdtime": mdtime,
        }
        for key, value in CREST_OPTIONS.items():
            if value:
                cmd_crest += f" --{key} {value}"

    workers = np.min([numThreads, n_confs])
    cpus_per_worker = numThreads // workers
    #args = [(xyz_file, cmd, cpus_per_worker) for xyz_file in xyz_files]
    args_crest = [(xyz_file, cmd_crest, cpus_per_worker, crest_version) for xyz_file in xyz_files]

### RUNING CREST FOR CONFORMATIONAL SAMPLING AND ENSEMBLE GENERATION ###
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        crest_results = executor.map(run_crest, args_crest) 
    
    for path, s in crest_results:
        path_confomer_ensemble = path
        crest_S_conf = s
#censo options
    CENSO_OPTIONS = {
        "input": path_confomer_ensemble,
        "solvent": solvent,
        "charge": charge,
        "balance": "on",
        "part0": "on",
        "thresholdpart0": 10.0,
        "part1": "on",
        "thresholdpart1": 6.0,
        "part2": "off",
        "thresholdpart2": 3.0,
        "maxthreads":numThreads,
        "omp":1,
        "balance":"on",
    }

    cmd_censo = ""
    for key, value in CENSO_OPTIONS.items():
        if value:
            cmd_censo += f" --{key} {value}"
    
    censo_destination=os.path.join(scr_dir, name)
    args_censo = [(path_confomer_ensemble, cmd_censo, cpus_per_worker, censo_destination)]
### RUNNING CENSO FOR CALCULATION OF BOLTZMANN AVERAGED MOLECULAR FREE ENERGY ###
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        censo_results = executor.map(run_censo, args_censo) 

    for e, path in censo_results:
        censo_free_energy = e
        censo_dirpath = path
# LogP calculation if required    
    LogP = 0
    if calc_LogP_OW:
        destination=os.path.join(scr_dir, name)
        LogP_path = os.path.join(destination, "LogP")
        os.makedirs(LogP_path)
        LogP_solvs = ["water","1-octanol"]
        G4LogP = []

        for LogP_solv in LogP_solvs: 
### RUNNING ORCA OPTIMIZATIONS WITH WATER AND 1-OCTANOL SOLVENTS FOR LogP CALCULATION ###
            args_LogP = [(censo_dirpath,LogP_solv,LogP_path,charge,cpus_per_worker)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                logp_results = executor.map(LogP_ow, args_LogP) 

            for e in logp_results:
                G4LogP.append(e)

        LogP = -(G4LogP[1] - G4LogP[0])*hartree2kcalmol/(2.303*R_gas*298.15)

    # Clean up
    if cleanup:
        shutil.rmtree(name)
    return censo_free_energy, crest_S_conf, LogP