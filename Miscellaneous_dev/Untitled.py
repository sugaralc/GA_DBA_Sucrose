#!/usr/bin/env python
# coding: utf-8

# In[1]:


from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
from rdkit.Chem import rdDepictor

#from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import random
import time
import sys

import crossover as co
import mutate as mu
import scoring_functions as sc
import xyz2mol as x2m
import os

charged_fragments = True
quick = True


# In[3]:


### Reading the directory to upload the molecules###
dir = os.getcwd()
xyzfiles = os.listdir('data_cmplxs/')
mols = []
for xyzfile in xyzfiles:
    file = "data_cmplxs/" + xyzfile
    atoms, charge_read, coordinates = x2m.read_xyz_file(file)
    raw_mol = x2m.xyz2mol(atoms, coordinates, charge=-2)
    mols.append(raw_mol[0])
    #print(raw_mol[0].GetNumAtoms())
    #print(Chem.MolToSmiles(raw_mol[0]))
    #print(Chem.MolToSmarts(raw_mol[0]))
    #Draw.MolToImage(raw_mol[0])
    #print(xyzfile)
########################################################

### Finding the bonds to remove the sucrose from the ligand ###
bonds_idxs = []
BC_idxs = []
for atom in mols[0].GetAtoms():
    if atom.GetAtomicNum() == 5:
        print("Boron atom:",atom.GetIdx())
        B_idx = atom.GetIdx()
        for neigh in atom.GetNeighbors():
            if neigh.GetAtomicNum() == 6:
                print("Neigh atom:",neigh.GetIdx())
                C_idx = neigh.GetIdx()
                bond = mols[0].GetBondBetweenAtoms(B_idx, C_idx)
                print("B_index, C_index, and B-C bond index:", B_idx, C_idx,bond.GetIdx())
                bonds_idxs.append(bond.GetIdx())
                BC_idxs.append([B_idx,C_idx])
                #create dictionary of indexes
#####################################################################

### Fragmenting the complex in the B-C bonds ###
print(bonds_idxs,BC_idxs)    
frags = Chem.FragmentOnBonds(mols[0], bonds_idxs)
frag1, frag2 = Chem.GetMolFrags(frags, asMols=True)
################################################

### Determining which one is the core fragment ###
for atom in frag1.GetAtoms():
    if atom.GetAtomicNum() == 5:
        core = frag1 
        linker = frag2
        break
    else:
        linker = frag1
        core = frag2
####################################################

#Draw.MolToImage(frags)
#MolsToGridImage(Chem.GetMolFrags(frags, asMols=True))

## Genetic operations here ##
mutated_ligand = mu.mutate(linker,0.9)

### Routine to combine the fragments after the genetic operations ###
cm = Chem.CombineMols(core,linker)

dummies = []
for atom in cm.GetAtoms():
    if atom.GetAtomicNum() == 0:
        dummy_idx = atom.GetIdx()
        dummy_label = atom.GetIsotope()
        for neigh in atom.GetNeighbors():
            dummy_idx_neigh = neigh.GetIdx()
            neigh_Z = neigh.GetAtomicNum()
        dummies.append([dummy_idx, dummy_label, dummy_idx_neigh, neigh_Z])

print(dummies)

new_bonds = []
dummy4BO3 = []
for BC_bond in BC_idxs:
    atom4bond = []
    label1 = BC_bond[0]
    label2 = BC_bond[1]
    for dummy in dummies:
        dummy_idx, dummy_label, dummy_idx_neigh, neigh_Z = dummy[0], dummy[1], dummy[2], dummy[3]
        if dummy_label == label1 or dummy_label == label2:
            atom4bond.append(dummy_idx_neigh)
            if len(atom4bond) == 2:
                print("all atoms in bond:",atom4bond)
                new_bonds.append(atom4bond)
        elif neigh_Z == 6:
            a4BO3 = "[" + str(dummy_label) + "*]"
            dummy4BO3.append(a4BO3)
            #print(a4BO3)

print("New bonds:",new_bonds)
em = Chem.RWMol(cm)

em.BeginBatchEdit()
for newbond in new_bonds:
    em.AddBond(newbond[0],newbond[1],Chem.rdchem.BondType.SINGLE)
for dummy in dummies:
    em.RemoveAtom(dummy[0])
em.CommitBatchEdit()

em.GetMol()
##############################################################################

#create the ligand with the boronic groups:
#print(Chem.MolToSmiles(linker))
#print(Chem.MolToSmarts(linker))

### Routine to generate the free ligand ###
lig_mod = Chem.ReplaceSubstructs(linker, 
                                 Chem.MolFromSmiles(dummy4BO3[0]), 
                                 Chem.MolFromSmiles('[B-](O)(O)(O)'),
                                 replaceAll=True)

lig = Chem.ReplaceSubstructs(lig_mod[0], 
                             Chem.MolFromSmiles(dummy4BO3[1]), 
                             Chem.MolFromSmiles('[B-](O)(O)(O)'),
                             replaceAll=True)
#############################################
print(Chem.MolToSmiles(lig[0]))
print(mutated_ligand)
print(Chem.MolToSmiles(mutated_ligand))

#print(cm,em)
#em.AddBond()
#Draw.MolToImage(mod_mol[0])

#rdDepictor.Compute2DCoords(lig[0])
#rdDepictor.Compute2DCoords(mutated_ligand)
em

