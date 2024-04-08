'''
Written by Gustavo Lara-Cruz 2024
'''

from rdkit import Chem

#from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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

def core_linker_frag(directory, xyzfile):
    """
    Function to separate the sucrose core from the linker, marking the extremes of how the linker is bonded to 
    sucrose. The dummy *98 is for the extreme of the linker bonded to the glucosyl side, and the dummy *99 is for the linker side
    bonded to the fructosyl side.
    """
    file = directory / xyzfile
    atoms, charge_read, coordinates = x2m.read_xyz_file(file)
    raw_mol = x2m.xyz2mol(atoms, coordinates, charge=chrg)
    mol = Chem.SanitizeMol(raw_mol[0])

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
            linker = frag2
            break
        else:
            linker = frag1
            core = frag2 
            
    suc_patt = list(frcyl_idxs + glcyl_idxs) 
    suc_patt.append(B_idx_glcyl)
    suc_patt.append(B_idx_frcyl)
    B_idxs = [B_idx_glcyl, B_idx_frcyl]
    for i in B_idxs:
        for neigh in mol.GetAtomWithIdx(i).GetNeighbors():
            if neigh.GetAtomicNum() == 8:
                if neigh.GetIdx() in suc_patt:
                    continue
                else:
                    suc_patt.append(neigh.GetIdx())
    core_coords = []
    suc_patt.sort()
    for idx in suc_patt:
        core_coords.append([mol.GetAtomWithIdx(idx).GetAtomicNum(),coordinates[idx]])
       
    return core,linker,core_coords

#def 
#    for xyzfile in xyzfiles:
#        file = directory / xyzfile
#        atoms, charge_read, coordinates = x2m.read_xyz_file(file)
#        chrg = -2
#        if xyzfile == 'cmplx113.xyz':
#            chrg = 0
#        elif xyzfile == 'cmplx114.xyz':
#            chrg = -1
#        raw_mol = x2m.xyz2mol(atoms, coordinates, charge=chrg)
#        Chem.SanitizeMol(raw_mol[0])
#        new_core, new_linkHs, core_xyz = core_linker_frag(raw_mol[0],coordinates)
#        new_link = Chem.RemoveHs(new_linkHs)
#
#        new_link.SetProp('pKaglcyl',str(8.8))
#        new_link.SetProp('pKafrcyl',str(8.8))
#    
#        linkers.append(new_link)
#        cores.append(new_core)
#        cmplxs.append(raw_mol[0])
#        cmplxs_xyz.append(coordinates)

#def linker_sanity_OK(mol,pos): #Functions already included in crossover.py as linker_OK
#    j = 0
#    if pos == 'middle':
#        for atom in mol.GetAtoms():
#            if atom.GetAtomicNum() == 0 and atom.GetIsotope == 96:
#                j += 1
#            if atom.GetAtomicNum() == 0 and atom.GetIsotope == 97:
#                j += 1
#        if j == 2:
#            return True
#        elif j < 2:
#            return False
#
#    elif pos == 'glcyl_side':
#        for atom in mol.GetAtoms():
#            if atom.GetAtomicNum() == 0 and atom.GetIsotope == 98:
#                j += 1
#            if atom.GetAtomicNum() == 0 and atom.GetIsotope == 97:
#                j += 1
#        if j == 2:
#            return True
#        elif j < 2:
#            return False
#
#    elif pos == 'frcyl_side':
#        for atom in mol.GetAtoms():
#            if atom.GetAtomicNum() == 0 and atom.GetIsotope == 99:
#                j += 1
#            if atom.GetAtomicNum() == 0 and atom.GetIsotope == 96:
#                j += 1
#        if j == 2:
#            return True
#        elif j < 2:
#            return False

def linker_crossover_OK(mol):
    i = 0 
    j = 0
    dummies = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() == 98:
            dummies.append(atom.GetIdx())
            #print('Atom idx and N:',atom.GetAtomicNum(),atom.GetIdx())
            for neigh in atom.GetNeighbors():
                if neigh.GetAtomicNum() == 6:
                    if neigh.GetIsAromatic() == True and neigh.IsInRing() == True:
                        i += 1
                        #print('Value of i =',i)
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() == 99:
            dummies.append(atom.GetIdx())
            #print('Atom idx and N:',atom.GetAtomicNum(),atom.GetIdx())
            for neigh in atom.GetNeighbors():
                if neigh.GetAtomicNum() == 6:
                    if neigh.GetIsAromatic() == True and neigh.IsInRing() == True:
                        j += 1
                        #print('Value of j =',j)

    #print('Dummies',len(dummies))
    if i == 1 and j == 1 and mol.GetNumAtoms() > 18 and len(dummies) == 2:
        return True
    else:
        return False

def tweezer_classification(mol,side):
    """
    Function to classify how the boronic groups are bonded to the linker. The types 0, 1, and 2 means that 
    the boronic groups are part of a PBA molecule in positions ortho, meta, or para, respectively. If the boronic
    group is directly bonded to the backbone of the linker, this side is classified as type 3.
    Input
    mol: Rdkit molecule with the dummy atoms as *98 and *99
    side: Strings "Glc" or "Frc" to indicate which side of the tweezer classify
    Output
    idx: 0, 1, 2, or 3
    : tuple list of atoms with the coincidences for the past
    """

    oPBA_glcyl_SMARTS = ['[#6!$([CX3]=[OX1])!$([C][CX3]=[OX1])!$([C][NX3]),#7!$([NX3;H2,O2])!$([N+]([O-])[O-])!$([#7][CX3]=[OX1]),#8!$([O][CX4;H3]),#16!H1!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1]1~[#6R1](-[98*])~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]1', \
                         '[#6!$([CX3]=[OX1])!$([C][CX3]=[OX1])!$([C][NX3]),#7!$([NX3;H2,O2])!$([N+]([O-])[O-])!$([#7][CX3]=[OX1]),#8!$([O][CX4;H3]),#16!H1!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1]1(-[98*])']
    mPBA_glcyl_SMARTS = ['[#6!$([CX3]=[OX1])!$([C][CX3]=[OX1])!$([C][NX3]),#7!$([NX3;H2,O2])!$([N+]([O-])[O-])!$([#7][CX3]=[OX1]),#8!$([O][CX4;H3]),#16!H1!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1]1~[#6R1,#7R1]~[#6R1](-[98*])~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]1', \
                         '[#6!$([CX3]=[OX1])!$([C][CX3]=[OX1])!$([C][NX3]),#7!$([NX3;H2,O2])!$([N+]([O-])[O-])!$([#7][CX3]=[OX1]),#8!$([O][CX4;H3]),#16!H1!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1](-[98*])~[#6R1,#7R1]1']
    pPBA_glcyl_SMARTS = ['[#6!$([CX3]=[OX1])!$([C][CX3]=[OX1])!$([C][NX3]),#7!$([NX3;H2,O2])!$([N+]([O-])[O-])!$([#7][CX3]=[OX1]),#8!$([O][CX4;H3]),#16!H1!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1](-[98*])~[#6R1,#7R1]~[#6R1,#7R1]1']
    fPBA_glcyl_SMARTS = ['[#6R3,#6R2,#7X3R2,#16R2]~[#6R2,#7X3R2,#16R2]~[#6R1](-[98*])', \
                         '[#6R3,#6R2,#7X3R2,#16R2]~[#6R3,#6R2,#7X3R2,#16R2]~[#6R1,#7R1,#16R1]~[#6R1](-[98*])']

    oPBA_frcyl_SMARTS = ['[#6!$([CX3]=[OX1])!$([C][CX3]=[OX1])!$([C][NX3]),#7!$([NX3;H2,O2])!$([N+]([O-])[O-])!$([#7][CX3]=[OX1]),#8!$([O][CX4;H3]),#16!H1!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1]1~[#6R1](-[99*])~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]1', \
                         '[#6!$([CX3]=[OX1])!$([C][CX3]=[OX1])!$([C][NX3]),#7!$([NX3;H2,O2])!$([N+]([O-])[O-])!$([#7][CX3]=[OX1]),#8!$([O][CX4;H3]),#16!H1!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1]1(-[99*])']
    mPBA_frcyl_SMARTS = ['[#6!$([CX3]=[OX1])!$([C][CX3]=[OX1])!$([C][NX3]),#7!$([NX3;H2,O2])!$([N+]([O-])[O-])!$([#7][CX3]=[OX1]),#8!$([O][CX4;H3]),#16!H1!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1]1~[#6R1,#7R1]~[#6R1](-[99*])~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]1', \
                         '[#6!$([CX3]=[OX1])!$([C][CX3]=[OX1])!$([C][NX3]),#7!$([NX3;H2,O2])!$([N+]([O-])[O-])!$([#7][CX3]=[OX1]),#8!$([O][CX4;H3]),#16!H1!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1](-[99*])~[#6R1,#7R1]1']
    pPBA_frcyl_SMARTS = ['[#6!$([CX3]=[OX1])!$([C][CX3]=[OX1])!$([C][NX3]),#7!$([NX3;H2,O2])!$([N+]([O-])[O-])!$([#7][CX3]=[OX1]),#8!$([O][CX4;H3]),#16!H1!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1](-[99*])~[#6R1,#7R1]~[#6R1,#7R1]1']
    fPBA_frcyl_SMARTS = ['[#6R3,#6R2,#7X3R2,#16R2]~[#6R2,#7X3R2,#16R2]~[#6R1](-[99*])', \
                         '[#6R3,#6R2,#7X3R2,#16R2]~[#6R3,#6R2,#7X3R2,#16R2]~[#6R1,#7R1,#16R1]~[#6R1](-[99*])']

    subs_glcyl_side = [oPBA_glcyl_SMARTS, mPBA_glcyl_SMARTS, pPBA_glcyl_SMARTS, fPBA_glcyl_SMARTS]
    subs_frcyl_side = [oPBA_frcyl_SMARTS, mPBA_frcyl_SMARTS, pPBA_frcyl_SMARTS, fPBA_frcyl_SMARTS]

    if side == 'Glc':
        for idx,sub_glcyl in enumerate(subs_glcyl_side):
            for smarts_sub in sub_glcyl:
                patt = Chem.MolFromSmarts(smarts_sub)
                #print('Glcyl side:',idx,mol.HasSubstructMatch(patt))
                if mol.HasSubstructMatch(patt):  
                    #print(mol.GetSubstructMatch(patt))
                    return idx, mol.GetSubstructMatch(patt)
    elif side == 'Frc':
        for idx,sub_frcyl in enumerate(subs_frcyl_side):
            for smarts_sub in sub_frcyl:
                patt = Chem.MolFromSmarts(smarts_sub)
                #print('Frcyl side:',idx,mol.HasSubstructMatch(patt))
                if mol.HasSubstructMatch(patt):
                    #print(mol.GetSubstructMatch(patt))
                    return idx, mol.GetSubstructMatch(patt)

if __name__ == "__main__":
    pass

