'''
Written by Gustavo Lara-Cruz 2024
'''

from rdkit import Chem
from rdkit.Chem import AllChem
#from rdkit.Geometry import Point3D
from rdkit.Chem import rdForceFieldHelpers

def linker_core_bonding(core,linker):
    """
    Function to bond the linker to sucrose taking into account the specific
    side of the linker bonded to the core. The dummy *98 is for the extreme of the 
    linker bonded to the glucosyl side, and the dummy *99 is for the linker side
    bonded to the fructosyl side.
    """
    cm = Chem.CombineMols(core,linker)
    dummies = []
    for atom in cm.GetAtoms():
        glcyl_side = 98
        frcyl_side = 99
        if atom.GetAtomicNum() == 0:
            dummies.append(atom.GetIdx())
            if atom.GetIsotope() == glcyl_side:
                dummies.append(atom.GetIdx())
                dummy_label = atom.GetIsotope()
                for neigh in atom.GetNeighbors():
                    if neigh.GetAtomicNum() == 6:
                        C_glc_side = neigh.GetIdx()
                    elif neigh.GetAtomicNum() == 5:
                        B_glc_side = neigh.GetIdx()
            elif atom.GetIsotope() == frcyl_side:
                dummies.append(atom.GetIdx())
                dummy_label = atom.GetIsotope()
                for neigh in atom.GetNeighbors():
                    if neigh.GetAtomicNum() == 6:
                        C_frc_side = neigh.GetIdx()
                    elif neigh.GetAtomicNum() == 5:
                        B_frc_side = neigh.GetIdx()                  
    em = Chem.RWMol(cm)
    em.BeginBatchEdit()
    em.AddBond(C_glc_side,B_glc_side,Chem.rdchem.BondType.SINGLE)
    em.AddBond(C_frc_side,B_frc_side,Chem.rdchem.BondType.SINGLE)
    for dummy in dummies:
        em.RemoveAtom(dummy)
    em.CommitBatchEdit()
    return em.GetMol()

def complex_conformer_generation(core,mol): 
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    cmplx_mut = linker_core_bonding(core, mol)
    cmplx_mut.UpdatePropertyCache(strict=False)
    cmplxh_mut = Chem.AddHs(cmplx_mut, addCoords=True)
    Chem.SanitizeMol(cmplxh_mut)

    suc_core_4patt = Chem.RWMol(core)
    suc_core_4patt.BeginBatchEdit()
    for atom in suc_core_4patt.GetAtoms():
        #print(atom.GetIdx(),atom.GetAtomicNum())
        if atom.GetAtomicNum() == 0:
            #print('Print dummy atom',atom.GetIdx())
            suc_core_4patt.RemoveAtom(atom.GetIdx())
    suc_core_4patt.CommitBatchEdit()
    Chem.SanitizeMol(suc_core_4patt)
    suc_core_mol = suc_core_4patt.GetMol()

    match_child = cmplxh_mut.GetSubstructMatch(suc_core_mol)

    cmap = {match_child[i]:suc_core_mol.GetConformer().GetAtomPosition(i) for i in range(len(match_child))}
    
    if AllChem.EmbedMolecule(cmplxh_mut,randomSeed=0xf00d,coordMap=cmap,useRandomCoords=True) > -1:
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(cmplxh_mut)
        for atidx in match_child:
            #print('atidx:',atidx)
            ff.UFFAddPositionConstraint(atidx,0.05,200)
        maxIters = 10
        while ff.Minimize(maxIts=1000) and maxIters>0:
            maxIters -= 1
        return cmplxh_mut
    elif AllChem.EmbedMolecule(cmplxh_mut,useRandomCoords=True) > -1:
        return cmplxh_mut
    else:
        return None
    
    #for i, atom in enumerate(cmplxh_mut.GetAtoms()):
    #    positions = cmplxh_mut.GetConformer().GetAtomPosition(i)
    #    print(atom.GetSymbol(), positions.x, positions.y, positions.z) 

def dummy2BO3(mol):
    """
    Function to convert the dummy atoms *98 and *99 into boronic groups
    """
    dummy_glc = "[" + str(98) + "*]"
    dummy_frc = "[" + str(99) + "*]"
    ligBO3 = Chem.ReplaceSubstructs(mol, 
                                    Chem.MolFromSmiles(dummy_glc), 
                                    Chem.MolFromSmiles('[B-](O)(O)(O)'),
                                    replaceAll=True)

    ligBO3_2 = Chem.ReplaceSubstructs(ligBO3[0], 
                                 Chem.MolFromSmiles(dummy_frc), 
                                 Chem.MolFromSmiles('[B-](O)(O)(O)'),
                                 replaceAll=True)
    return ligBO3_2[0]

def free_ligand_conformer_generation(mol):
    free_ligand = dummy2BO3(mol) 
    free_ligand.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(free_ligand)
    free_ligandH = Chem.AddHs(free_ligand, addCoords=True)
    free_ligandH.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(free_ligandH)

    if AllChem.EmbedMolecule(free_ligandH,randomSeed=0xf00d,useRandomCoords=True) > -1:
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(free_ligandH)
        maxIters = 10
        while ff.Minimize(maxIts=1000) and maxIters>0:
            maxIters -= 1
        return free_ligandH
    elif AllChem.EmbedMolecule(free_ligandH,useRandomCoords=True) > -1:
        return free_ligandH
    else:
        return None

def conformer_generation(mol):
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    molH = Chem.AddHs(mol, addCoords=True)
    molH.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(molH)

    if AllChem.EmbedMolecule(molH,randomSeed=0xf00d,useRandomCoords=True) > -1:
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(molH)
        maxIters = 10
        while ff.Minimize(maxIts=1000) and maxIters>0:
            maxIters -= 1
        return molH
    elif AllChem.EmbedMolecule(free_ligandH,useRandomCoords=True) > -1:
        return molH
    else:
        return None

    #for i, atom in enumerate(free_ligandH.GetAtoms()):
    #    positions = free_ligandH.GetConformer().GetAtomPosition(i)
    #    print(atom.GetSymbol(), positions.x, positions.y, positions.z) 

if __name__ == "__main__":
    linker_smi = '[98*]C1=CC=CC2=C1CC1=CC3=C(C=C1C2)CCC=C3C1=CC([99*])=CC=C1'
    linker = Chem.MolFromSmiles(linker_smi)
    #complex_conformer_generation(linker)
    free_ligand_conformer_generation(linker)
