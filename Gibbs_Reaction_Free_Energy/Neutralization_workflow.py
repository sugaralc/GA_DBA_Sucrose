#!/usr/bin/env python
# coding: utf-8

from rdkit import Chem#, DataStructs
#from rdkit.Chem.Fraggle import FraggleSim
#from rdkit.Chem import Draw
#from rdkit.Chem import AllChem
#from rdkit.Chem import Descriptors
#from rdkit.Chem import rdmolops
#from rdkit.Chem import rdDepictor
#from rdkit import RDConfig
#from rdkit.Chem import rdFingerprintGenerator
#from rdkit.Geometry import Point3D
#from random import randrange
import copy

#from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os

from neutralize import neutralize_smiles

def Tweezer_neutralization(smile):
    neutral_smiles1 = neutralize_smiles([smile])
    mol2 = Chem.MolFromSmiles(neutral_smiles1[0])
    if Chem.GetFormalCharge(mol2) != 0:
        rings_info = mol2.GetRingInfo()
        B_smarts = Chem.MolFromSmarts("[#5-]")
        N_smarts = Chem.MolFromSmarts("[#7R1+]")
        N_matchs = mol2.GetSubstructMatches(N_smarts)
        BC_pairs = []
        B_matches = []
        for B_match in mol2.GetSubstructMatches(B_smarts):
            B_atom = mol2.GetAtomWithIdx(B_match[0])
            B_matches.append(B_match[0])
            for neigh in B_atom.GetNeighbors():
                if neigh.GetAtomicNum() == 6:
                    BC_pairs.append([B_match[0],neigh.GetIdx()])
        BExclude = []
        for BC_pair in BC_pairs:
            for N_match in N_matchs:
                if rings_info.AreAtomsInSameRing(BC_pair[1],N_match[0]):
                    BExclude.append(BC_pair[0])
        BO_groups = []
        B_matches.sort(reverse=True)
        for B_match in B_matches:
            if B_match not in BExclude:
                B_atom = mol2.GetAtomWithIdx(B_match)
                O_atoms = []
                for neigh in B_atom.GetNeighbors():
                    if neigh.GetAtomicNum() == 8:
                        O_atoms.append(neigh.GetIdx())
                O_atoms.sort(reverse=True)
                BO_groups.append(O_atoms)
                        
        em = Chem.RWMol(mol2)
        em.BeginBatchEdit()
        i = 0
        for B_match in B_matches:
            if B_match not in BExclude:
                Batom = em.GetAtomWithIdx(B_match)
                Batom.SetFormalCharge(0)
                em.RemoveAtom(BO_groups[i][0])
                i += 1
        em.CommitBatchEdit()
        mol3 = em.GetMol()
        if Chem.GetFormalCharge(mol3) != 0:
            while Chem.GetFormalCharge(mol3) != 0:
                Cl_anion = Chem.MolFromSmiles("[Cl-]")
                Na_cation = Chem.MolFromSmiles("[Na+]")
                if Chem.GetFormalCharge(mol3) < 0:
                    molfrag = Chem.CombineMols(mol3,Na_cation)
                elif Chem.GetFormalCharge(mol3) > 0:
                    molfrag = Chem.CombineMols(mol3,Cl_anion)
                mol3 = copy.deepcopy(molfrag)    
            return mol3
        else:
            return mol3
    else:
        return mol2