import copy
import os
import sys

import numpy as np
from rdkit import Chem

free_energy_dir = os.path.dirname(__file__)
sys.path.append(free_energy_dir)

from make_structures import complex_conformer_generation, free_ligand_conformer_generation
from utils import hartree2kcalmol
from CREST_CENSO_utils import molecular_free_energy

suc_core_file = os.path.join(free_energy_dir, "Sucrose_core_dummies.mol")
suc_core = Chem.MolFromMolFile(suc_core_file, removeHs=False, sanitize=True)

frag_energies = np.sum([-8.232710038092, -19.734652802142, -32.543971411432])  # 34 atoms


def free_energy_scoring(linker, idx=(0, 0), ncpus=1, n_confs=10, cleanup=False):
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
    complex_3d = complex_conformer_generation(
        mol=linker,
        core=suc_core#,
        #numConfs=n_confs,
        #pruneRmsThresh=0.1,
        #force_constant=1e12,
    )

    # Embed free ligand. Here Perform the conformer generation for the free ligand.
    free_ligand_3d = free_ligand_conformer_generation(
        mol=linker#,
        #numConfs=n_confs,
        #pruneRmsThresh=0.1,
        #force_constant=1e12,
    )

    energy = molecular_free_energy(complex_3d)

    # Calc Energy of TS
    ts3d_energy, ts3d_geom = xtb_optimize(
        ts3d,
        gbsa="methanol",
        opt_level="tight",
        name=f"{idx[0]:03d}_{idx[1]:03d}_ts",
        input=os.path.join(catalyst_dir, "input_files/constr.inp"),
        numThreads=ncpus,
        cleanup=cleanup,
    )

    # Embed Catalyst
    cat3d = copy.deepcopy(cat)
    cat3d = Chem.AddHs(cat3d)
    cids = Chem.rdDistGeom.EmbedMultipleConfs(cat3d, numConfs=n_confs, pruneRmsThresh=0.1)
    if len(cids) == 0:
        raise ValueError(f"Could not embed catalyst {Chem.MolToSmiles(Chem.RemoveHs(cat))}")

    # Calc Energy of Cat
    cat3d_energy, cat3d_geom = xtb_optimize(
        cat3d,
        gbsa="methanol",
        opt_level="tight",
        name=f"{idx[0]:03d}_{idx[1]:03d}_cat",
        numThreads=ncpus,
        cleanup=cleanup,
    )

    # Calculate electronic activation energy
    De = (ts3d_energy - frag_energies - cat3d_energy) * hartree2kcalmol
    return De, ts3d_geom, cat3d_geom

if __name__ == "__main__":
    linker_smi = '[98*]C1=CC=CC2=C1CC1=CC3=C(C=C1C2)CCC=C3C1=CC([99*])=CC=C1'
    linker = Chem.MolFromSmiles(linker_smi)
    #complex_conformer_generation(linker)
    free_energy_scoring(linker)