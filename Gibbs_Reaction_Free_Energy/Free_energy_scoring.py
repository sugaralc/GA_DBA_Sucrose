import copy
import os
import sys

import numpy as np
from rdkit import Chem

free_energy_dir = os.path.dirname(__file__)
sys.path.append(free_energy_dir)

from make_structures import constrained_conformer_generation, free_ligand_conformer_generation, conformer_generation
from utils import hartree2kcalmol
from CREST_CENSO_utils import molecular_free_energy

suc_core_file = os.path.join(free_energy_dir, "Sucrose_core_dummies.mol")
suc_core = Chem.MolFromMolFile(suc_core_file, removeHs=False, sanitize=True)
sucrose_S_conf = (12.513341 +  9.805273 + 12.519072)/3 #Sconf in cal/mol.K units. Values from 80 ps for sampling
                                                       #using the -entropy keyword from crest.
sucrose_free_energy = ((-1297.5253819)+(-1297.5252573)+(-1297.5257083))/3 #Molecular free energies from censo part1 and r2scan-3c
                                                                          #Values from 80 ps for metadynamics sampling.
cluster4H2O_free_energy = ((-305.6843641)+(-305.6843772)+(-305.6843547))/3 #Molecular free energies from censo part1 and r2scan-3c
                                                                            #Values from 80 ps of metadynamics sampling.
R_gas = 1.985877534/1000 #kcal/mol.T units
Temp = 298.15 #K
H2ONormalConcentration = 55.34 #Normal concentration of water

def free_energy_scoring(linker, idx=(0, 0), ncpus=1, cleanup=False):
    """Calculates reaction free energy in kcal/mol between products(DBA-Suc + 4H_2O) and 
    reactants (DBA + Suc)

    Args:
        linker (rdkit.Mol): Molecule containing one tertiary amine
        n_confs (int, optional): Nubmer of confomers used for embedding. Defaults to 10.
        cleanup (bool, optional): Clean up files after calculation.
                                  Defaults to False, needs to be False to work with submitit.

    Returns:
        Tuple: Contains energy difference, Geom of complex and Geom of free ligand
    """

    # Embed Complex. Here Perform the conformer generation for the complex. 
    # Pending, add the options used for the conformer generation.
    complex_3d = constrained_conformer_generation(
        mol=linker,
        core=suc_core,
        sucrose=True,
        #numConfs=n_confs,
        #pruneRmsThresh=0.1,
        #force_constant=1e12,
    )

    # Embed free ligand. Here Perform the conformer generation for the free ligand.
    tweezer_3d = free_ligand_conformer_generation(
        mol=linker#,
        #numConfs=n_confs,
        #pruneRmsThresh=0.1,
        #force_constant=1e12,
    )
    #tweezer_3d = conformer_generation(mol=linker)
    complex_free_energy, complex_S_conf, _ = molecular_free_energy(
        complex_3d,
        solvent="h2o",
        smiles=None,
        name=f"{idx[0]:03d}_{idx[1]:03d}_complex",
        calc_LogP_OW=False,
        #input=os.path.join(catalyst_dir, "input_files/constr.inp"),
        numThreads=ncpus,
        cleanup=cleanup,
    )

    tweezer_free_energy, tweezer_S_conf, LogP = molecular_free_energy(
        tweezer_3d,
        solvent="h2o",
        smiles=linker,
        name=f"{idx[0]:03d}_{idx[1]:03d}_tweezer",
        calc_LogP_OW=True,
        #input=os.path.join(catalyst_dir, "input_files/constr.inp"),
        numThreads=ncpus,
        cleanup=cleanup,
    )

    # Calculate reaction free energy including conformational entropy
    ds = (complex_S_conf - tweezer_S_conf - sucrose_S_conf)*Temp/1000
    cluster4H2O_Gcorrected = cluster4H2O_free_energy - R_gas*Temp/hartree2kcalmol*np.log(H2ONormalConcentration/4) 
    dg = (complex_free_energy + cluster4H2O_Gcorrected  - tweezer_free_energy - sucrose_free_energy) * hartree2kcalmol
    DG = dg - ds
    
    # Add the function to calculate the solvation free energy difference between octanol 
    #and water using the censo best conformer for the free ligand. 

    print("Reaction Free Energy (kcal/mol) =",DG)
    print("LogP_o/w =",LogP)

    if DG < -15.0:
        DG = -15.0
    elif DG > 5.0:
        DG = 1.0

    if LogP < -3.0:
        LogP = -3.0
    elif LogP > 6.0:
        LogP = 6.0

    return DG, LogP

if __name__ == "__main__":
    #linker_smi = '[98*]C1=CC=CC2=C1CC1=CC3=C(C=C1C2)CCC=C3C1=CC([99*])=CC=C1'
    #linker_smi = '[98*]C1=C([C@H]2CC=CC3=C2CC2=C3CC3=C([99*])C=CC=C3C2)C=CN=C1F'
    #linker_smi = '[98*]C1=CC=CC([C@@H]2CC[C@]3(CC[C@H](C4=CC([99*])=CC=C4)C(=C)C3)C2)=C1'
    #linker_smi = '[98*]C1=CC=CC=C1[C@@H]1CC[C@@]2(C=C(C3=CC([99*])=CC=C3)CC2)C1'
    linker_smi = '[98*]c1ccc([C@@H]2C=CC[C@]3(CC[C@@H](c4cccc([99*])c4)C(=C)C3)C2)cc1'
    #linker_smi = 'CCCC'
    linker = Chem.MolFromSmiles(linker_smi)
    #complex_conformer_generation(linker)
    free_energy_scoring(linker,ncpus=38,idx=(0,7))
