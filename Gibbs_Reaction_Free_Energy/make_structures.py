'''
Written by Gustavo Lara-Cruz 2024
'''

from rdkit import Chem
from rdkit.Chem import AllChem
#from rdkit.Geometry import Point3D
from rdkit.Chem import rdForceFieldHelpers
import copy

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

def constrained_conformer_generation(core,mol,sucrose=True): 
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)

    #cmplx_mut = linker_core_bonding(core, mol)
    #cmplx_mut.UpdatePropertyCache(strict=False)
    #cmplxh_mut = Chem.AddHs(cmplx_mut, addCoords=True)
    #Chem.SanitizeMol(cmplxh_mut)

    if sucrose:
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
    else:
        suc_core_mol = Chem.AddHs(core, addCoords=True)
        cmplx_mut = mol
        cmplxh_mut = Chem.AddHs(cmplx_mut, addCoords=True)
        Chem.SanitizeMol(suc_core_mol)
        Chem.SanitizeMol(cmplxh_mut)

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

def BO32H(mol):
    """
    Function to convert the dummy atoms *98 and *99 into boronic groups
    """
    #ligBO2 = Chem.ReplaceSubstructs(mol, 
    #                                Chem.MolFromSmarts('[B-](O)(O)(O)'), 
    #                                Chem.MolFromSmiles('B'),
    #                                replaceAll=True)

    ligBO2 = AllChem.DeleteSubstructs(mol, 
                                 Chem.MolFromSmarts('[B-](O)(O)(O)'))
    return ligBO2

def BCbond_4frag(atoms_idx,mol): 
    """
    Function to get the B-C bond index.
    """
    for idx in atoms_idx:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 8:
            for neigh in atom.GetNeighbors():
                if neigh.GetAtomicNum() == 5:
                    B_idx = neigh.GetIdx()
                    for B_neigh in neigh.GetNeighbors():
                        if B_neigh.GetAtomicNum() == 6:
                            C_idx = B_neigh.GetIdx()
                            bond = mol.GetBondBetweenAtoms(B_idx, C_idx)
                            return bond.GetIdx(), B_idx

def core_linker_frag(mol):
    """
    Function to separate the sucrose core from the linker, marking the extremes of how the linker is bonded to 
    sucrose. The dummy *98 is for the extreme of the linker bonded to the glucosyl side, and the dummy *99 is for the linker side
    bonded to the fructosyl side.
    """

    frcyl = Chem.MolFromSmarts('C1(O)C(O)C(CO)OC1(CO)')    
    frcyl_idxs = mol.GetSubstructMatch(frcyl)
    glcyl = Chem.MolFromSmarts('C1C(O)C(O)C(CO)OC1(O)')
    glcyl_idxs = mol.GetSubstructMatch(glcyl)
    Glcyl_dLabel = 98
    BC_glcyl, B_idx_glcyl = BCbond_4frag(glcyl_idxs,mol) #Getting B-C bond index in glc side 
    Frcyl_dLabel = 99
    BC_frcyl, B_idx_frcyl = BCbond_4frag(frcyl_idxs,mol) #Getting B-C bond index in frc side 

    frags = Chem.FragmentOnBonds(mol, [BC_glcyl,BC_frcyl], dummyLabels=[(Glcyl_dLabel,Glcyl_dLabel),(Frcyl_dLabel,Frcyl_dLabel)])
    frag1, frag2 = Chem.GetMolFrags(frags, asMols=True)
    for atom in frag1.GetAtoms():
        if atom.GetAtomicNum() == 5:
            core = frag1 
            linkerH = frag2
            break
        else:
            linkerH = frag1
            core = frag2 
    linker = Chem.RemoveHs(linkerH)

    return linker

#def free_ligand_conformer_generation(mol):
#    free_ligand = dummy2BO3(mol) 
#    free_ligand.UpdatePropertyCache(strict=False)
#    Chem.SanitizeMol(free_ligand)
#    free_ligandH = Chem.AddHs(free_ligand, addCoords=True)
#    free_ligandH.UpdatePropertyCache(strict=False)
#    Chem.SanitizeMol(free_ligandH)
#
#    if AllChem.EmbedMolecule(free_ligandH,randomSeed=0xf00d,useRandomCoords=True) > -1:
#        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(free_ligandH)
#        maxIters = 10
#        while ff.Minimize(maxIts=1000) and maxIters>0:
#            maxIters -= 1
#        return free_ligandH
#    elif AllChem.EmbedMolecule(free_ligandH,useRandomCoords=True) > -1:
#        return free_ligandH
#    else:
#        return None

def free_ligand_conformer_generation(mol):
    core = core_linker_frag(mol)

    dummy_atoms = []
    for atom in core.GetAtoms():
        if atom.GetAtomicNum() == 0:
            dummy_atoms.append(atom.GetIdx())

    free_ligand = dummy2BO3(core) 
    free_ligandH = Chem.AddHs(free_ligand, addCoords=True)
    free_ligandH.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(free_ligandH)

    em = Chem.RWMol(core)
    em.BeginBatchEdit()
    for dummy in dummy_atoms:
        em.RemoveAtom(dummy)
    em.CommitBatchEdit()
    template = em.GetMol()

    match_child = free_ligandH.GetSubstructMatch(template)
    cmap = {match_child[i]:template.GetConformer().GetAtomPosition(i) for i in range(len(match_child))}

    if AllChem.EmbedMolecule(free_ligandH,randomSeed=0xf00d,coordMap=cmap,useRandomCoords=True) > -1:
        ff = rdForceFieldHelpers.UFFGetMoleculeForceField(free_ligandH)
        for atidx in match_child:
            #print('atidx:',atidx)
            ff.UFFAddPositionConstraint(atidx,0.05,200)
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
    #linker_smi = '[98*]C1=CC=CC2=C1CC1=CC3=C(C=C1C2)CCC=C3C1=CC([99*])=CC=C1.[Na+]'
    #linker_smi = "[98*]c1c([H])c([H])c([H])c(C2=C([H])C([H])([H])c3c([H])c(-c4c([H])c([H])c([H])c([H])c4[99*])c([H])c([H])c32)c1[H].[Na+].[Na+]"
    linker_smi = '[H]N=C(c1c([H])c([H])c([98*])c([H])[n+]1C([H])([H])C(=O)O[H])[C@@]1([H])Oc2c([H])c([H])c([H])c([H])c2C([H])([H])C1=C([H])C([H])=C(C#C[n+]1c([H])nc([H])c([99*])c1[H])C([H])([H])[H]'
    linker = Chem.MolFromSmiles(linker_smi)
    linker = Chem.RemoveHs(linker)
    lnk_smarts = Chem.MolFromSmarts(Chem.MolToSmarts(linker))
    print(Chem.MolToSmiles(linker))
    #for i, atom in enumerate(linker.GetAtoms()):
    #    positions = free_ligandH.GetConformer().GetAtomPosition(i)
        #print(atom.GetSymbol(),atom.GetIdx()) 
    #complex_conformer_generation(linker)
    import xyz2mol as x2m
    #import os
    #print(os.getcwd())
    atoms, charge_read, coordinates = x2m.read_xyz_file("./Gibbs_Reaction_Free_Energy/cmplx_from_censo.xyz")
    raw_mol = x2m.xyz2mol(atoms, coordinates, charge=charge_read)
    core = raw_mol[0]
    core = Chem.RemoveHs(core)
    for i, atom in enumerate(core.GetAtoms()):
        positions = core.GetConformer().GetAtomPosition(i)
        print(atom.GetSymbol(), positions.x, positions.y, positions.z) 

    #free_ligandH = free_ligand_conformer_generation(linker)
    free_ligandH = free_ligand_conformer_generation(core)
    #free_ligandH = constrained_conformer_generation(core,linker,sucrose=False,free_tweezer=True)

    print("free ligand")
    for i, atom in enumerate(free_ligandH.GetAtoms()):
        positions = free_ligandH.GetConformer().GetAtomPosition(i)
        print(atom.GetSymbol(), positions.x, positions.y, positions.z) 
