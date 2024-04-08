'''
Written by Gustavo Lara-Cruz 2024
'''

from rdkit import Chem 

#from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import sys

import crossover as co
import mutate as mu
import PBA_mutation as PBA_mu
import os

sys.path.insert(1, 'MolGpKa/src')
import protonate 

from pKA_similarity import pKaFromSimmilarity
from DBA_GA_utils import tweezer_classification

co.average_size = 110.15 #I should define this variable in other place
co.size_stdev = 3.50     #I should define this variable in other place

Glcyl_dLabel = 96
Frcyl_dLabel = 97


def PBAGlcyl_linker_PBAFrcyl(linker,glcylpatt,frcylpatt):
    #print("Two bond breakings and three fragments")
    glcyl_a1, glcyl_a2 = glcylpatt[:2]
    frcyl_a1, frcyl_a2 = frcylpatt[:2]
    
    #print('Atoms Glcyl side:',glcyl_a1, glcyl_a2)
    #print('Atoms Frcyl side:',frcyl_a1, frcyl_a2)
    
    glcyl_bond = linker.GetBondBetweenAtoms(glcyl_a1, glcyl_a2)
    frcyl_bond = linker.GetBondBetweenAtoms(frcyl_a1, frcyl_a2)
    
    frags = Chem.FragmentOnBonds(linker, [glcyl_bond.GetIdx(),frcyl_bond.GetIdx()], \
                                dummyLabels=[(Glcyl_dLabel,Glcyl_dLabel),(Frcyl_dLabel,Frcyl_dLabel)])
    
    #print('After FragOnBonds:',Chem.MolToSmiles(frags))
    
    frag1, frag2, frag3 = Chem.GetMolFrags(frags, asMols=True)
    #print('Raw fragments:',Chem.MolToSmiles(frag1),Chem.MolToSmiles(frag2),Chem.MolToSmiles(frag3))

    
    for atom in frag1.GetAtoms():
        #print(atom.GetAtomicNum(), atom.GetIsotope(),atom.GetIdx())
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() == 98:
            #print('frag1 is PBAglc4mut')
            PBAglc4mut = frag1 
            break
        elif atom.GetAtomicNum() == 0 and atom.GetIsotope() == 99:
            #print('frag1 is PBAfrc4mut')
            PBAfrc4mut = frag1
            break
        elif atom.GetAtomicNum() == 0 and atom.GetIsotope() == 96:
            for atom2 in frag1.GetAtoms():
                if atom2.GetAtomicNum() == 0 and atom2.GetIsotope() == 97:
                    #print('frag1 is linker4mut')
                    linker4mut = frag1
    
    for atom in frag2.GetAtoms():
        #print(atom.GetAtomicNum(), atom.GetIsotope(),atom.GetIdx())
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() == 98:
            #print('frag2 is PBAglc4mut')
            PBAglc4mut = frag2 
            break
        elif atom.GetAtomicNum() == 0 and atom.GetIsotope() == 99:
            #print('frag2 is PBAfrc4mut')
            PBAfrc4mut = frag2
            break
        elif atom.GetAtomicNum() == 0 and atom.GetIsotope() == 96:
            for atom2 in frag2.GetAtoms():
                if atom2.GetAtomicNum() == 0 and atom2.GetIsotope() == 97:
                    #print('frag2 is linker4mut')
                    linker4mut = frag2

    for atom in frag3.GetAtoms():
        #print(atom.GetAtomicNum(), atom.GetIsotope(),atom.GetIdx())
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() == 98:
            #print('frag3 is PBAglc4mut')
            PBAglc4mut = frag3 
            break
        elif atom.GetAtomicNum() == 0 and atom.GetIsotope() == 99:
            #print('frag3 is PBAfrc4mut')
            PBAfrc4mut = frag3
            break
        elif atom.GetAtomicNum() == 0 and atom.GetIsotope() == 96:
            for atom2 in frag1.GetAtoms():
                if atom2.GetAtomicNum() == 0 and atom2.GetIsotope() == 97:
                    #print('frag3 is linker4mut')
                    linker4mut = frag3
            
    #print('Fragments classification:',Chem.MolToSmiles(PBAglc4mut),Chem.MolToSmiles(PBAfrc4mut),Chem.MolToSmiles(linker4mut))  
            
    p = [4/6, 1/6, 1/6] 
    mut_operation = np.random.choice(['linker', 'PBAglcyl', 'PBAfrcyl'], p=p)
    if mut_operation == 'linker':
        #print("Do mutation on the linker")
        linker_mut = mu.mutate(linker4mut,1.0,'middle')
        linker_mut_smi = Chem.MolToSmiles(linker_mut, kekuleSmiles=True)
        if linker.HasProp('pKaglcyl') ==1 and linker.HasProp('pKafrcyl') == 1:
            PBAglc_child, pKaglcyl_child = PBAglc4mut, linker.GetProp('pKaglcyl')
            PBAfrc_child, pKafrcyl_child = PBAfrc4mut, linker.GetProp('pKafrcyl')
        else:
            PBAglc_child = PBAglc4mut
            PBAglc4mut_smi = Chem.MolToSmiles(PBAglc4mut, kekuleSmiles=True)
            pKaglcyl_child = pKaFromSimmilarity(PBAglc4mut_smi)
            PBAfrc_child = PBAfrc4mut
            PBAfrc4mut_smi = Chem.MolToSmiles(PBAfrc4mut, kekuleSmiles=True)
            pKafrcyl_child = pKaFromSimmilarity(PBAfrc4mut_smi)
    elif mut_operation == 'PBAglcyl':
        #print("Do mutation on the PBAglcyl side")
        linker_mut = linker4mut
        linker_mut_smi = Chem.MolToSmiles(linker_mut, kekuleSmiles=True)
        PBAglc_child, pKaglcyl_child = PBA_mu.Glcyl()
        if linker.HasProp('pKafrcyl') == 1:
            PBAfrc_child, pKafrcyl_child = PBAfrc4mut, linker.GetProp('pKafrcyl')
        else:
            PBAfrc_child = PBAfrc4mut
            PBAfrc4mut_smi = Chem.MolToSmiles(PBAfrc4mut, kekuleSmiles=True)
            pKafrcyl_child = pKaFromSimmilarity(PBAfrc4mut_smi)
    elif mut_operation == 'PBAfrcyl':
        #print("Do mutation on the PBAfrcyl side")
        linker_mut = linker4mut
        linker_mut_smi = Chem.MolToSmiles(linker_mut, kekuleSmiles=True)
        if linker.HasProp('pKaglcyl') ==1:
            PBAglc_child, pKaglcyl_child = PBAglc4mut, linker.GetProp('pKaglcyl')
        else:
            PBAglc_child = PBAglc4mut
            PBAglc4mut_smi = Chem.MolToSmiles(PBAglc4mut, kekuleSmiles=True)
            pKaglcyl_child = pKaFromSimmilarity(PBAglc4mut_smi)
        PBAfrc_child, pKafrcyl_child = PBA_mu.Frcyl() 

    if float(pKaglcyl_child) > float(pKafrcyl_child):
        pH = float(pKaglcyl_child) + 1.0
        linker_child_smi = protonate.protonate_mol(linker_mut_smi,pH,0.0)
        linker_child = Chem.MolFromSmiles(linker_child_smi[0])
    else:
        pH = float(pKafrcyl_child) + 1.0
        linker_child_smi = protonate.protonate_mol(linker_mut_smi,pH,0.0)
        linker_child = Chem.MolFromSmiles(linker_child_smi[0])
    
    cm1 = Chem.CombineMols(PBAglc_child, PBAfrc_child)
    cm = Chem.CombineMols(cm1, linker_child)
    #print(Chem.MolToSmiles(cm))
    dummies = []
    atoms_glc_side = []
    atoms_frc_side = []
    for atom in cm.GetAtoms():
        glcyl_side = 96
        frcyl_side = 97
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() not in [98, 99]:
            dummies.append(atom.GetIdx())
            if atom.GetIsotope() == glcyl_side:
                dummies.append(atom.GetIdx())
                for neigh in atom.GetNeighbors():
                    atoms_glc_side.append(neigh.GetIdx()) 
            elif atom.GetIsotope() == frcyl_side:
                dummies.append(atom.GetIdx())     
                for neigh in atom.GetNeighbors():
                    atoms_frc_side.append(neigh.GetIdx()) 
                    
    em = Chem.RWMol(cm)
    em.BeginBatchEdit()
    em.AddBond(atoms_glc_side[0],atoms_glc_side[1],Chem.rdchem.BondType.SINGLE)
    em.AddBond(atoms_frc_side[0],atoms_frc_side[1],Chem.rdchem.BondType.SINGLE)
    for dummy in dummies:
        em.RemoveAtom(dummy)
    em.CommitBatchEdit()
    em.SetProp('pKaglcyl',str(pKaglcyl_child))
    em.SetProp('pKafrcyl',str(pKafrcyl_child))

    return em.GetMol()

def PBAGlcyl_linkerPBAFrcyl(linker,glcylpatt):
    glcyl_a1, glcyl_a2 = glcylpatt[:2]
    glcyl_bond = linker.GetBondBetweenAtoms(glcyl_a1, glcyl_a2)
    frags = Chem.FragmentOnBonds(linker, [glcyl_bond.GetIdx()], \
                                 dummyLabels=[(Glcyl_dLabel,Glcyl_dLabel)])

    #print('After FragOnBonds:',Chem.MolToSmiles(frags))
        
    frag1, frag2 = Chem.GetMolFrags(frags, asMols=True)
    for atom in frag1.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() == 98:
            PBAglc4mut = frag1 
            linker4mut = frag2
            break
        else:
            linker4mut = frag1
            PBAglc4mut = frag2
        
    #print("Breaking on glcyl side")
    p = [4/6, 2/6]
    mut_operation = np.random.choice(['linker', 'PBAglcyl'], p=p)    
    if mut_operation == 'linker':
        #print("Do mutation on the linker")
        linker_mut = mu.mutate(linker4mut,1.0,'frcyl')
        linker_mut_smi = Chem.MolToSmiles(linker_mut, kekuleSmiles=True)
        pKafrcyl_child = pKaFromSimmilarity(linker_mut_smi)
        if linker.HasProp('pKaglcyl') ==1:
            PBAglc_child, pKaglcyl_child = PBAglc4mut, linker.GetProp('pKaglcyl')
        else:
            PBAglc_child = PBAglc4mut
            PBAglc4mut_smi = Chem.MolToSmiles(PBAglc4mut, kekuleSmiles=True)
            pKaglcyl_child = pKaFromSimmilarity(PBAglc4mut_smi)
    elif mut_operation == 'PBAglcyl':
        #print("Do mutation on the PBAglcyl side")
        linker_mut = linker4mut
        linker_mut_smi = Chem.MolToSmiles(linker_mut, kekuleSmiles=True)
        if linker.HasProp('pKafrcyl') == 1:
            pKafrcyl_child = linker.GetProp('pKafrcyl')
        else:
            pKafrcyl_child = pKaFromSimmilarity(linker_mut_smi)
        PBAglc_child,pKaglcyl_child = PBA_mu.Glcyl() 
        

    if float(pKaglcyl_child) > float(pKafrcyl_child):
        pH = float(pKaglcyl_child) + 1.0
        linker_child_smi = protonate.protonate_mol(linker_mut_smi,pH,0.0)
        linker_child = Chem.MolFromSmiles(linker_child_smi[0])
    else:
        pH = float(pKafrcyl_child) + 1.0
        linker_child_smi = protonate.protonate_mol(linker_mut_smi,pH,0.0)
        linker_child = Chem.MolFromSmiles(linker_child_smi[0])
        
        
    cm = Chem.CombineMols(PBAglc_child, linker_child)
    dummies = []
    atoms_glc_side = []
    for atom in cm.GetAtoms():
        glcyl_side = 96
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() not in [98, 99]:
            dummies.append(atom.GetIdx())
            if atom.GetIsotope() == glcyl_side:
                dummies.append(atom.GetIdx())
                for neigh in atom.GetNeighbors():
                    atoms_glc_side.append(neigh.GetIdx()) 
                    
    em = Chem.RWMol(cm)
    em.BeginBatchEdit()
    em.AddBond(atoms_glc_side[0],atoms_glc_side[1],Chem.rdchem.BondType.SINGLE)
    for dummy in dummies:
        em.RemoveAtom(dummy)
    em.CommitBatchEdit()
    em.SetProp('pKaglcyl',str(pKaglcyl_child))
    em.SetProp('pKafrcyl',str(pKafrcyl_child))
    return em.GetMol()
    
def PBAGlcyllinker_PBAfrcyl(linker,frcylpatt):
    frcyl_a1, frcyl_a2 = frcylpatt[:2]
    frcyl_bond = linker.GetBondBetweenAtoms(frcyl_a1, frcyl_a2)
    frags = Chem.FragmentOnBonds(linker, [frcyl_bond.GetIdx()], \
                                 dummyLabels=[(Frcyl_dLabel,Frcyl_dLabel)])

    #print('After FragOnBonds:',Chem.MolToSmiles(frags))
        
    frag1, frag2 = Chem.GetMolFrags(frags, asMols=True)
    for atom in frag1.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() == 99:
            PBAfrc4mut = frag1 
            linker4mut = frag2
            break
        else:
            linker4mut = frag1
            PBAfrc4mut = frag2
                
    #print("Breaking on frcyl side")
    p = [4/6, 2/6]
    mut_operation = np.random.choice(['linker', 'PBAfrcyl'], p=p)
    if mut_operation == 'linker':
        #print("Do mutation on the linker")
        linker_mut = mu.mutate(linker4mut,1.0,'glcyl')
        linker_mut_smi = Chem.MolToSmiles(linker_mut, kekuleSmiles=True)
        pKaglcyl_child = pKaFromSimmilarity(linker_mut_smi)
        if linker.HasProp('pKafrcyl') ==1:
            PBAfrc_child, pKafrcyl_child = PBAfrc4mut, linker.GetProp('pKafrcyl')
        else:
            PBAfrc_child = PBAfrc4mut
            PBAfrc4mut_smi = Chem.MolToSmiles(PBAfrc4mut, kekuleSmiles=True)
            pKafrcyl_child = pKaFromSimmilarity(PBAfrc4mut_smi)
    elif mut_operation == 'PBAfrcyl':
        #print("Do mutation on the PBAfrcyl side")
        linker_mut = linker4mut
        linker_mut_smi = Chem.MolToSmiles(linker_mut, kekuleSmiles=True)
        if linker.HasProp('pKaglcyl') == 1:
            pKaglcyl_child = linker.GetProp('pKaglcyl')
        else:
            pKaglcyl_child = pKaFromSimmilarity(linker_mut_smi)
        PBAfrc_child, pKafrcyl_child = PBA_mu.Frcyl() 

    if float(pKaglcyl_child) > float(pKafrcyl_child):
        pH = float(pKaglcyl_child) + 1.0
        linker_child_smi = protonate.protonate_mol(linker_mut_smi,pH,0.0)
        linker_child = Chem.MolFromSmiles(linker_child_smi[0])
    else:
        pH = float(pKafrcyl_child) + 1.0
        linker_child_smi = protonate.protonate_mol(linker_mut_smi,pH,0.0)
        linker_child = Chem.MolFromSmiles(linker_child_smi[0])
                
    cm = Chem.CombineMols(PBAfrc_child, linker_child)
    dummies = []
    atoms_frc_side = []
    for atom in cm.GetAtoms():
        frcyl_side = 97
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() not in [98, 99]:
            dummies.append(atom.GetIdx())
            if atom.GetIsotope() == frcyl_side:
                dummies.append(atom.GetIdx())     
                for neigh in atom.GetNeighbors():
                    atoms_frc_side.append(neigh.GetIdx()) 
                    
    em = Chem.RWMol(cm)
    em.BeginBatchEdit()
    em.AddBond(atoms_frc_side[0],atoms_frc_side[1],Chem.rdchem.BondType.SINGLE)
    for dummy in dummies:
        em.RemoveAtom(dummy)
    em.CommitBatchEdit()
    em.SetProp('pKaglcyl',str(pKaglcyl_child))
    em.SetProp('pKafrcyl',str(pKafrcyl_child))
    return em.GetMol()

def PBA_mutation(linker,idx_glcyl,glcylpatt,idx_frcyl,frcylpatt):

    if idx_glcyl in [0,1,2] and idx_frcyl in [0,1,2]:
        return(PBAGlcyl_linker_PBAFrcyl(linker,glcylpatt,frcylpatt))
    elif idx_glcyl in [0,1,2] and idx_frcyl == 3:
        return(PBAGlcyl_linkerPBAFrcyl(linker,glcylpatt))
    elif idx_frcyl in [0,1,2] and idx_glcyl == 3:
        return(PBAGlcyllinker_PBAfrcyl(linker,frcylpatt))
    else:
        return None

if __name__ == "__main__":

    #smiles1 = '[98*]C1=CC=CC2=C1C1=CC3=C(C=CC(C4=CC([99*])=CC=C4)=C3)C=C1CC2' #Frycl side fused
    #smiles1 = '[98*]C1=CC=CC([C@H]2C=C[C@]3(CC[C@@H](C4=CC([99*])=CC=C4)CC3)C2)=C1' #Linker whitout fusion
    smiles1 = '[98*]C1=CC=CC([C@@H]2C=CC3=C(C=C4CC5=C(C=CC=C5[99*])C4=C3)C2)=C1' #Glcyl side fused
    
    mol1 = Chem.MolFromSmiles(smiles1)
    mol1.SetProp('pKaglcyl',str(8.8))
    mol1.SetProp('pKafrcyl',str(8.8))
    Chem.SanitizeMol(mol1)
    idx_glcyl, glcylpatt = tweezer_classification(mol1,'Glc')
    idx_frcyl, frcylpatt = tweezer_classification(mol1,'Frc')

    mutated_child = PBA_mutation(mol1,idx_glcyl,glcylpatt,idx_frcyl,frcylpatt)
    print(Chem.MolToSmiles(mutated_child,kekuleSmiles=True))
    print('pKaglcyl =',mutated_child.GetProp('pKaglcyl'))
    print('pKafrcyl =',mutated_child.GetProp('pKafrcyl'))
