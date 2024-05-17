'''
Written by Gustavo Lara-Cruz 2024
'''

from rdkit import Chem

#from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import xyz2mol as x2m

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

def dummy2BO2(mol):
    """
    Function to convert the dummy atoms *98 and *99 into boronic groups
    """
    dummy_glc = "[" + str(98) + "*]"
    dummy_frc = "[" + str(99) + "*]"
    ligBO3 = Chem.ReplaceSubstructs(mol, 
                                    Chem.MolFromSmiles(dummy_glc), 
                                    Chem.MolFromSmiles('B(O)(O)'),
                                    replaceAll=True)

    ligBO3_2 = Chem.ReplaceSubstructs(ligBO3[0], 
                                 Chem.MolFromSmiles(dummy_frc), 
                                 Chem.MolFromSmiles('B(O)(O)'),
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
    raw_mol = x2m.xyz2mol(atoms, coordinates, charge=charge_read)
    Chem.SanitizeMol(raw_mol[0])
    mol = raw_mol[0]
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
    
    if linker.HasProp('pKaglcyl') == 0:
        linker.SetProp('pKaglcyl',str(8.8))
    if linker.HasProp('pKafrcyl') == 0:        
        linker.SetProp('pKafrcyl',str(8.8))
#    suc_patt = list(frcyl_idxs + glcyl_idxs) 
#    suc_patt.append(B_idx_glcyl)
#    suc_patt.append(B_idx_frcyl)
#    B_idxs = [B_idx_glcyl, B_idx_frcyl]
#    for i in B_idxs:
#        for neigh in mol.GetAtomWithIdx(i).GetNeighbors():
#            if neigh.GetAtomicNum() == 8:
#                if neigh.GetIdx() in suc_patt:
#                    continue
#                else:
#                    suc_patt.append(neigh.GetIdx())
#    core_coords = []
#    suc_patt.sort()
#    for idx in suc_patt:
#        core_coords.append([mol.GetAtomWithIdx(idx).GetAtomicNum(),coordinates[idx]])
       
#    return core,linker,core_coords
    return linker

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
    mol.UpdatePropertyCache()
    #Chem.GetSymmSSSR(mol)
    #print("Mol to classify:",Chem.MolToSmiles(mol))
    
    oPBA_glcyl_SMARTS = ['[#6!H3!$([#6X4](F)(F)F)!$([#6X3]=[#8X1])!$([#6!R][#6X3!R](=[#8X1])[H1,#8X2H,#6X4])!$([#6!R][#6X3!R&H1]=[#8])!$([#6!R][#7X3H2!R,#7X4+]),#7!$([#7X3;H2,O2])!$([N+]([O-])(=O))!$([#7][#6X3]=[#8X1])!$([#7!R][#6!R][#6X3](=[#8X1])[#8]),#8!$([#8][#6X4&H3]),#16!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1+]1~[#6R1](-[98*])~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]1', \
                         '[#6!H3!$([#6X4](F)(F)F)!$([#6X3]=[#8X1])!$([#6!R][#6X3!R](=[#8X1])[H1,#8X2H,#6X4])!$([#6!R][#6X3!R&H1]=[#8])!$([#6!R][#7X3H2!R,#7X4+]),#7!$([#7X3;H2,O2])!$([N+]([O-])(=O))!$([#7][#6X3]=[#8X1])!$([#7!R][#6!R][#6X3](=[#8X1])[#8]),#8!$([#8][#6X4&H3]),#16!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1+]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1]1(-[98*])']
    mPBA_glcyl_SMARTS = ['[#6!H3!$([#6X4](F)(F)F)!$([#6X3]=[#8X1])!$([#6!R][#6X3!R](=[#8X1])[H1,#8X2H,#6X4])!$([#6!R][#6X3!R&H1]=[#8])!$([#6!R][#7X3H2!R]),#7!$([#7X3;H2,O2])!$([N+]([O-])(=O))!$([#7][#6X3]=[#8X1])!$([#7!R][#6!R][#6X3](=[#8X1])[#8]),#8!$([#8][#6X4&H3]),#16!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1+]1~[#6R1,#7R1]~[#6R1](-[98*])~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]1', \
                         '[#6!H3!$([#6X4](F)(F)F)!$([#6X3]=[#8X1])!$([#6!R][#6X3!R](=[#8X1])[H1,#8X2H,#6X4])!$([#6!R][#6X3!R&H1]=[#8])!$([#6!R][#7X3H2!R]),#7!$([#7X3;H2,O2])!$([N+]([O-])(=O))!$([#7][#6X3]=[#8X1])!$([#7!R][#6!R][#6X3](=[#8X1])[#8]),#8!$([#8][#6X4&H3]),#16!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1+]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1](-[98*])~[#6R1,#7R1]1']
    pPBA_glcyl_SMARTS = ['[#6!H3!$([#6X4](F)(F)F)!$([#6X3]=[#8X1])!$([#6!R][#6X3!R](=[#8X1])[H1,#8X2H,#6X4])!$([#6!R][#6X3!R&H1]=[#8])!$([#6!R][#7X3H2!R]),#7!$([#7X3;H2,O2])!$([N+]([O-])(=O))!$([#7][#6X3]=[#8X1])!$([#7!R][#6!R][#6X3](=[#8X1])[#8]),#8!$([#8][#6X4H&3]),#16!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1+]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1](-[98*])~[#6R1,#7R1]~[#6R1,#7R1]1']
    fPBA_glcyl_SMARTS = ['[#6R3,#6R2,#7X3R2,#16R2]~[#6R2,#7X3R2,#16R2]~[#6R1](-[98*])', \
                         '[#6R3,#6R2,#7X3R2,#16R2]~[#6R3,#6R2,#7X3R2,#16R2]~[#6R1,#7R1,#16R1]~[#6R1](-[98*])']

    oPBA_frcyl_SMARTS = ['[#6!H3!$([#6X4](F)(F)F)!$([#6X3]=[#8X1])!$([#6!R][#6X3!R](=[#8X1])[H1,#8X2H,#6X4H3])!$([#6!R][#6X3!R&H1]=[#8])!$([#6!R][#7X3H2!R,NX4+]),#7!$([#7X3;H2,O2])!$([N+]([O-])(=O))!$([#7][#6X3]=[#8X1])!$([#7!R][#6!R][#6X3](=[#8X1])[#8]),#8!$([O][CX4;H3]),#16!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1+]1~[#6R1](-[99*])~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]1', \
                         '[#6!H3!$([#6X4](F)(F)F)!$([#6X3]=[#8X1])!$([#6!R][#6X3!R](=[#8X1])[H1,#8X2H,#6X4H3])!$([#6!R][#6X3!R&H1]=[#8])!$([#6!R][#7X3H2!R,NX4+]),#7!$([#7X3;H2,O2])!$([N+]([O-])(=O))!$([#7][#6X3]=[#8X1])!$([#7!R][#6!R][#6X3](=[#8X1])[#8]),#8!$([O][CX4;H3]),#16!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1+]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1]1(-[99*])']
    mPBA_frcyl_SMARTS = ['[#6!H3!$([#6X4](F)(F)F)!$([#6X3]=[#8X1])!$([#6!R][#6X3!R](=[#8X1])[H1,#8X2H,#6X4H3])!$([#6!R][#6X3!R&H1]=[#8])!$([#6!R][#7X3H2!R]),#7!$([#7X3;H2,O2])!$([N+]([O-])(=O))!$([#7][#6X3]=[#8X1])!$([#7!R][#6!R][#6X3](=[#8X1])[#8]),#8!$([O][CX4;H3]),#16!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1+]1~[#6R1,#7R1]~[#6R1](-[99*])~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]1', \
                         '[#6!H3!$([#6X4](F)(F)F)!$([#6X3]=[#8X1])!$([#6!R][#6X3!R](=[#8X1])[H1,#8X2H,#6X4H3])!$([#6!R][#6X3!R&H1]=[#8])!$([#6!R][#7X3H2!R]),#7!$([#7X3;H2,O2])!$([N+]([O-])(=O))!$([#7][#6X3]=[#8X1])!$([#7!R][#6!R][#6X3](=[#8X1])[#8]),#8!$([O][CX4;H3]),#16!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1+]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1](-[99*])~[#6R1,#7R1]1']
    pPBA_frcyl_SMARTS = ['[#6!H3!$([#6X4](F)(F)F)!$([#6X3]=[#8X1])!$([#6!R][#6X3!R](=[#8X1])[H1,#8X2H,#6X4H3])!$([#6!R][#6X3!R&H1]=[#8])!$([#6!R][#7X3H2!R]),#7!$([#7X3;H2,O2])!$([N+]([O-])(=O))!$([#7][#6X3]=[#8X1])!$([#7!R][#6!R][#6X3](=[#8X1])[#8]),#8!$([O][CX4;H3]),#16!$([#16X4](=[OX1])(=[OX1]))]~[#6R1,#7R1+]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1](-[99*])~[#6R1,#7R1]~[#6R1,#7R1]1']
    fPBA_frcyl_SMARTS = ['[#6R3,#6R2,#7X3R2,#16R2]~[#6R2,#7X3R2,#16R2]~[#6R1](-[99*])', \
                         '[#6R3,#6R2,#7X3R2,#16R2]~[#6R3,#6R2,#7X3R2,#16R2]~[#6R1,#7R1,#16R1]~[#6R1](-[99*])']

    mPBA_glcyl_SMARTS2 = ['[98*]-[#6R1]1~[#6R1,#7R1]~[#6R1,#7R1,#7+R1](-[#6])~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]1']
                              #'[98*]-[#6R1]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1,#7+](-[#6!$([CX3]=[OX1])!$([C][CX3]=[OX1])!$([C][NX3])])~[#6R1,#7R1]1']
                              #'[#6!H3]-[#7+]-1=[#6]-[#6](-[98*])=[#6]-[#6]=[#6]-1']
    pPBA_glcyl_SMARTS2 = ['[98*]-[#6R1]1~[#6R1,#7R1]~[#6R1,#7R1][#6R1,#7R1,#7+R1](-[#6])~[#6R1,#7R1]~[#6R1,#7R1]1' ,\
                          '[#6]-[#7+]-1=[#6]-[#6]=[#6](-[98*])-[#6]=[#6]-1']

    pPBA_frcyl_SMARTS2 = ['[99*]-[#6R1]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1+](-[#6!H3!$([#6X4](F)(F)F)!$([CX3]=[OX1])!$([C][CX3!R]=[OX1])!$([C][NX3!R])])~[#6R1,#7R1]~[#6R1,#7R1]1']
    #                          '[99*]-[#6R1]1~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1]~[#6R1,#7R1,#7+](-[#6!$([CX3]=[OX1])!$([C][CX3]=[OX1])!$([C][NX3])])~[#6R1,#7R1]1', \
    #                          '[#6!H3]-[#7+0]-1=[#6]-[#6](-[99*])=[#6]-[#6]=[#6]-1']

    subs_glcyl_side = [oPBA_glcyl_SMARTS, mPBA_glcyl_SMARTS, pPBA_glcyl_SMARTS, fPBA_glcyl_SMARTS]#, mPBA_glcyl_SMARTS2, pPBA_glcyl_SMARTS2]
    subs_frcyl_side = [oPBA_frcyl_SMARTS, mPBA_frcyl_SMARTS, pPBA_frcyl_SMARTS, fPBA_frcyl_SMARTS]#, pPBA_frcyl_SMARTS2]

    if side == 'Glc':
        for idx,sub_glcyl in enumerate(subs_glcyl_side):
            for smarts_sub in sub_glcyl:
                patt = Chem.MolFromSmarts(smarts_sub)
                #patt.UpdatePropertyCache()
                #print('Glcyl side:',idx,mol.HasSubstructMatch(patt))
                smiles = Chem.MolToSmiles(mol)
                if mol.HasSubstructMatch(patt):  
                    #print(mol.GetSubstructMatch(patt))
                    if idx == 4:
                        print("Pyridum Glcyl",idx,smiles,Chem.MolToSmiles(patt))
                        idx = 1
                    if idx == 5:
                        print("Pyridum Glcyl",idx,smiles,Chem.MolToSmiles(patt))
                        idx = 2
                    return idx, mol.GetSubstructMatch(patt)
    elif side == 'Frc':
        for idx,sub_frcyl in enumerate(subs_frcyl_side):
            for smarts_sub in sub_frcyl:
                patt = Chem.MolFromSmarts(smarts_sub)
                #patt.UpdatePropertyCache()
                #print('Frcyl side:',idx,mol.HasSubstructMatch(patt))
                smiles = Chem.MolToSmiles(mol)
                if mol.HasSubstructMatch(patt):
                    if idx == 4:
                        print("Pyridium Frcyl",smiles,Chem.MolToSmiles(patt))
                        idx = 3
                    #print(mol.GetSubstructMatch(patt))
                    return idx, mol.GetSubstructMatch(patt)

if __name__ == "__main__":

    smiles = ['[98*]C1=CC=CC2=C1CC1=CC3=C(C=C1C2)CCC=C3C1=CC([99*])=CC=C1',
              '[98*]C1=C([C@H]2CC=CC3=C2CC2=C3CC3=C([99*])C=CC=C3C2)C=CN=C1F',
              '[98*]C1=CC=CC([C@@H]2CC[C@]3(CC[C@H](C4=CC([99*])=CC=C4)C(=C)C3)C2)=C1',
              '[98*]C1=CC=CC=C1[C@@H]1CC[C@@]2(C=C(C3=CC([99*])=CC=C3)CC2)C1',
              '[98*]c1ccc([C@@H]2C=CC[C@]3(CC[C@@H](c4cccc([99*])c4)C(=C)C3)C2)cc1',
              '[98*]C1=CC=CC([C@@H]2CC[C@]3(CC[C@H](C4=CC([99*])=CC=C4)C(=C)C3)C2)=C1',
              '[98*]C1=CC=CC([C@H]2CCC3=C(CC4=CC5=C(C=C34)C([99*])=CC=C5)C2)=C1',
              '[98*]C1=CC=CC(/C=C2/CC[C@]3(CC[C@@H](C4=CC([99*])=CC=C4)CC3)C2)=C1',
              '[98*]C1=CC=CC(/C=C2/CCC[C@]23CC[C@@H](C2=CC([99*])=CC=C2)CC3)=C1',
              '[98*]C1=CC=CC=C1C1=CC2=C(C(C3=CC([99*])=CC=C3)=CC=C2)C2=C1C=CC1=C2C=CC=C1',
              '[98*]C1=CC=CC([C@@H]2CC[C@]3(CC[C@H](C4=CC([99*])=CC=C4)CC3)C2)=C1',
              '[98*]C1=CC=CC2=C1C1=CC3=C(C=CC(C4=CC([99*])=CC=C4)=C3)C=C1CC2',
              '[98*]C1=CC=CC([C@@H]2CC3=C(C=C4CC5=C(C=CC=C5[99*])CC4=C3)C2)=C1',
              '[98*]C1=CC=CC=C1C1=CC2=C(C=C1)C1=C(C=CC=C1C1=CC([99*])=CC=C1)C2',
              '[98*]C1=CC=CC2=C1CC1=C3C[C@H](C4=CC([99*])=CC=C4)CC3=CC=C1C2',
              '[98*]C1=CC=CC([C@H]2C3=C(C=CC=C3)C3=C2C=CC(C2=CC([99*])=CC=C2)=C3)=C1',
              '[98*]C1=CC=CC([C@@H]2C=CC3=C(CC4=C3CC3=C4C([99*])=CC=C3)C2)=C1',
              '[98*]C1=CC=CC([C@H]2CC3=C(C=C4C[C@H](C5=CC([99*])=CC=C5)CC4=C3)C2)=C1',
              '[98*]C1=CC=CC2=C1C1=C(C=CC3=C1C=CC(C1=CC([99*])=CC=C1)=C3)CC2',
              '[98*]C1=CC=CC([C@@H]2CCC[C@H]([C@@H]3CCC[C@H](C4=CC([99*])=CC=C4)C3)C2)=C1',
              '[98*]C1=CC=CC([C@H]2C=CC3=CC(C4=CC([99*])=CC=C4)=CC=C32)=C1',
              '[98*]C1=CC=CC([C@H]2CCC3=C2C=CC2=CC4=C(C=C32)C([99*])=CC=C4)=C1',
              '[98*]C1=CC=CC([C@H]2CC[C@]3(C=CC[C@H](C4=CC([99*])=CC=C4)C3)CC2)=C1',
              '[98*]C1=CC=CC=C1C1=CCC2=C(C3=CC([99*])=CC=C3)C=CC3=CC=CC1=C23',
              '[98*]C1=CC=CC=C1C1=C2CC=CC3=CC=C(C4=CC([99*])=CC=C4)C(=C32)C=C1',
              '[98*]C1=CC=CC([C@@H]2C=CC3=C4CC5=C(C([99*])=CC=C5)C4=CC=C3C2)=C1',
              '[98*]C1=CC=CC(C2=CC=CC3=C2C2=C(C=C(C4=CC([99*])=CC=C4)C=C2)C3)=C1',
              '[98*]C1=CC=CC(C2=CC3=C(C=C2)CC2=C3C=C(C3=CC([99*])=CC=C3)C=C2)=C1',
              '[98*]C1=CC=CC2=C1CC1=CC3=C(C=C1C2)CCC=C3C1=CC([99*])=CC=C1',
              '[98*]C1=CC=CC([C@@H]2C[C@@H]3CC[C@@H](C4=CC([99*])=CC=C4)C[C@@H]3C2)=C1',
              '[98*]C1=CC=CC=C1C1=CC2=C(C=C1)C(C1=CC([99*])=CC=C1)=C1C=CC=CC1=C2',
              '[98*]C1=CC=CC([C@@H]2C=CC3=C(C=C4C=CC5=C(C=CC=C5[99*])C4=C3)C2)=C1',
              '[98*]C1=CC=CC([C@H]2CC[C@@H](CC3=CC([99*])=CC=C3)C[C@H]2C)=C1',
              '[98*]C1=CC=CC(C2=CC3=C(C=C2)C=CC2=C3C3=C(C=C2)C(C2=CC([99*])=CC=C2)=CC=C3)=C1',
              '[98*]C1=CC=CC(C2=CC=CC3=C2C2=C(C=C3)C(C3=CC([99*])=CC=C3)=CC=C2)=C1',
              '[98*]C1=CC=CC([C@@H]2CC3=C(C=C4CC5=C(C=CC=C5[99*])CC4=C3)C2)=C1',
              '[98*]C1=CC=CC([C@@H]2C=CC3=C(C2)C2=C(C=C3)C3=C(C=C2)C([99*])=CC=C3)=C1',
              '[98*]C1=CC=CC2=C1C=C1C3=C(CC[C@@H](C4=CC([99*])=CC=C4)C3)CC1=C2',
              '[98*]C1=CC=CC=C1C1=CC2=C(CC1)CC1=C2CC2=C1C=CC=C2[99*]',
              '[98*]C1=CC=CC=C1C1=CC=CC2=C1CC1=C2C(C2=CC([99*])=CC=C2)=CC=C1',
              '[98*]C1=CC=C(CCC2=CCC3=C(C[N+]4=CC=C([99*])C=C4)C=CC4=CC=CC2=C43)C=C1C=O',
              '[98*]c1cnccc1[C@@H]1Cc2cc3c(cc2C1=O)Cc1c([99*])cccc1C3',
              '[98*]c1cccc([S@@H]2C=Cc3c(ccc4c3Cc3cccc([99*])c3-4)C2)c1',
              '[98*]C1=CC=CC([C@@H]2CCC[C@H]([C@@H]3CCC(=O)[C@@H](C4=CC([99*])=CC=C4)C3)C2)=C1',
              '[98*]C1=CC=CC([SH]2CC[C@]23CC[C@H](C2=CC([99*])=CC=C2)C(=C)C3)=C1',
              '[98*]C1=CC=CC([C@@H](C)CC[C@]2(C)C=CC[C@H](C[N+]3=CC=CC([99*])=C3F)C2)=C1',
              '[98*]C1=CC=[N+](C[C@@H]2C=CC3=C(CC4=C3CC3=CC=CC([99*])=C34)C2)C=C1',
              '[98*]C1=CC=C(C(=O)NCCC)[N+](C[C@H]2CC=CC3=C2CC2=C3CC3=C([99*])C=CC=C3C2)=C1',
              '[98*]C1=CC=C[N+](CCC2=CC=CC3=CC(C[N+]4=CN=CC([99*])=C4)C(C(O)C#N)=C32)=C1',
              '[98*]C1=CC=C(C(=O)NCCC)[N+](CC2=CC3=C(CC2)CC2=C3CC3=C([99*])C=CC=C32)=C1',
              '[98*]C1=CC=C(C(=O)NCCC)[N+](C[C@@H]2CC[C@]3(CC[C@@H](C4=NC=C([99*])C=N4)CC3)C2)=C1',
              '[98*]C1=CC=C(Br)C=C1[C@@H]1CC[C@@]2(C=C(C3=CC([99*])=CC=C3)C(=O)C2)C1',
              '[98*]C1=CC=C([C@H](C#C)NC[C@H]2CC[C@H](SC3=C([99*])C(F)=CC=C3[N+](=O)[O-])C(=N)C2)C(NC(=O)C=C)=C1',
              '[98*]C1=CC=CC([C@]2(F)CC3=CC4=C(C=C3C2)C[C@H](C[N+]2=CC([99*])=CC=C2C(=O)O)C4)=C1',
              '[98*]C1=CC=CC([C@@H]2CC3=CC4=C(C=C3CN2)CC2=C([99*])C=CC=C2C4)=C1',
              '[98*]C1=CC=C(C2=CC(Cl)=C3CC4=C(C3=C2)C(Br)=C(C[N+]2=CC([99*])=CC=C2C(=O)NCCC)C=C4)C(C(F)(F)F)=C1',
              '[98*]C1=C([C@@H]2C=CC3=CC=C4C5=CC=CC([99*])=C5C=CC4=C3C2=O)C=CN=C1F',
              '[98*]C1=C(C=O)C=CC2=C1CC1=C3C[C@H](C[N+]4=CC([99*])=CC=C4C(=O)O)CC3=CC=C1C2',
              '[98*]C1=CC=CC([C@@H]2C=CC3=CC=C4C5=CC=CC([99*])=C5C=CC4=C3C2=O)=C1',
              '[98*]C1=C(C(F)(F)F)C=CC2=C1C1=CC3=CC(C[N+]4=CC([99*])=CC=C4C(=O)O)=CC=C3C=C1CC2',
              '[98*]C1=CC=C(C(=O)NCCC)[N+](C[C@H]2CC[C@@H](CC3=NC(C(=O)O)=CC=C3[99*])C[C@H]2C)=C1',
              '[98*]C1=C(C=O)C=CC2=C1CC1=C3C[C@H](C[N+]4=CC([99*])=CC=C4C(=O)O)CC3=CC=C1C2',
              '[98*]C1=CC=C(C(=O)NCCC)[N+](C[C@@H]2CC[C@]3(CC[C@@H](C4=NC=C([99*])C=N4)CC3)C2)=C1',
              '[98*]C1=CC=C(CCC2=CCC3=C(C[N+]4=CC=C([99*])C=C4)C=CC4=CC=CC2=C43)C=C1C=O',
              '[98*]C1=CC=C(C(=O)NCCC)[N+](CC2=CC3=C(CC2)CC2=C3CC3=C([99*])C=CC=C32)=C1',
              '[98*]C1=CC=C(CCC2=CCC3=C(C[N+]4=CC=C([99*])C=C4)C=CC4=CC=CC2=C43)C=C1C=O',
              'C[NH+](C)Cc1ccc(cc1[99*])-c1ccc2cccc3C(=CCc1c23)c1ccccc1[98*]'
              #'[98*]C1=C(F)C(=[N+]([O-])[O-])/C(=[S+]/[H])C([C@@H]2C=CC3=C(C=C4CC5=C(C(F)=C(N)C=C5[99*])C4=C3C(=O)O)C2)=C1',
              ]
    molecules = [Chem.MolFromSmiles(mol) for mol in smiles]
    for idx,mol in enumerate(molecules):
        print('Glcy side',idx)
        idx_glcyl, patt_glcyl = tweezer_classification(mol,'Glc')
        print('idx_glcyl=',idx_glcyl,'patt_glcyl=',patt_glcyl)
        print('Frcyl side',idx)
        idx_frcyl, patt_frcyl = tweezer_classification(mol,'Frc')
        print('idx_frcyl=',idx_frcyl,'patt_frcyl=',patt_frcyl)
    

