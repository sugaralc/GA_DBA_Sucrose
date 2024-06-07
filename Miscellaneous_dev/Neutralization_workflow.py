#!/usr/bin/env python
# coding: utf-8

from rdkit import Chem, DataStructs
from rdkit.Chem.Fraggle import FraggleSim
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.Chem.Draw import IPythonConsole, MolsToGridImage
from rdkit.Chem import rdDepictor
from rdkit import RDConfig
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Geometry import Point3D
from rdkit.Chem import rdForceFieldHelpers
from random import randrange
import py3Dmol
import copy

#from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import xyz2mol as x2m
import os

from neutralize import neutralize_smiles

charged_fragments = True
quick = True

def drawit(ms, p=None, confId=-1, removeHs=True,colors=('cyanCarbon','redCarbon','blueCarbon')):
        if p is None:
            p = py3Dmol.view(width=600, height=600)
        p.removeAllModels()
        for i,m in enumerate(ms):
            if removeHs:
                m = Chem.RemoveHs(m)
            IPythonConsole.addMolToView(m,p,confId=confId)
            p.setStyle({'model':-1,},
                            {'stick':{'colorscheme':colors[i%len(colors)]}})
        p.zoomTo()
        return p.show()

def show_3dmol(image_id=None,inchi=None,smiles=None,mol=None):
    mol = mol
    if not mol:
        print('No molecule is provided')
        return
           
    molh = Chem.AddHs(mol)
    #if AllChem.EmbedMolecule(molh,randomSeed=0xf00d)<0:
    #    print('Failed to embed in 3d')
    #    return
    pdb_data = Chem.MolToPDBBlock(molh)
    view = py3Dmol.view(width=600, height=600, query=None, data=pdb_data, linked=False)
    view.setStyle({'model':-1,},{'stick': {}})
    #view.setBackgroundColor('#f9f4fb')
    return view

with open("smiles", "r+") as file1:
    # Reading from a file
    smiles = [line.rstrip() for line in file1]
    #print(len(smiles))
    for smile in smiles:
        mol1 = Chem.MolFromSmiles(smile)
        if Chem.GetFormalCharge(mol1) != 0:
            neutral_smiles1 = neutralize_smiles([Chem.MolToSmiles(mol1)])
            mol2 = Chem.MolFromSmiles(neutral_smiles1[0])
            if Chem.GetFormalCharge(mol2) != 0:
                rings = mol2.GetRingInfo().AtomRings()
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
                    print("Returning neutral mol by adding counter anions")
                else:
                    print("Returning neutral mol mol3")
            else:
                print("Returning neutral mol mol2")       
        else:
            print("Returning neutral mol mol1")
