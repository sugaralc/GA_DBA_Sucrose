'''
Written by Jan H. Jensen 2018
'''
from rdkit import Chem
from rdkit.Chem import AllChem

import random
import numpy as np
import crossover4DBA as co

from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')

def delete_atom():
  choices = ['[*:1]~[D1]>>[*:1]', '[*:1]~[D2]~[*:2]>>[*:1]-[*:2]',
             '[*:1]~[D3](~[*;!H0:2])~[*:3]>>[*:1]-[*:2]-[*:3]',
             '[*:1]~[D4](~[*;!H0:2])(~[*;!H0:3])~[*:4]>>[*:1]-[*:2]-[*:3]-[*:4]',
             '[*:1]~[D4](~[*;!H0;!H1:2])(~[*:3])~[*:4]>>[*:1]-[*:2](-[*:3])-[*:4]']
  p = [0.25,0.25,0.25,0.1875,0.0625]
  
  return np.random.choice(choices, p=p)

def append_atom():
  choices = [['single',['C','N','O','F','S','Cl','Br'],7*[1.0/7.0]],
             ['double',['C','N','O'],3*[1.0/3.0]],
             ['triple',['C','N'],2*[1.0/2.0]] ]
  p_BO = [0.60,0.35,0.05]
  
  index = np.random.choice(list(range(3)), p=p_BO)
  
  BO, atom_list, p = choices[index]
  new_atom = np.random.choice(atom_list, p=p)
  
  if BO == 'single': 
    rxn_smarts = '[*;!H0:1]>>[*:1]X'.replace('X','-'+new_atom)
  if BO == 'double': 
    rxn_smarts = '[*;!H0;!H1:1]>>[*:1]X'.replace('X','='+new_atom)
  if BO == 'triple': 
    rxn_smarts = '[*;H3:1]>>[*:1]X'.replace('X','#'+new_atom)
    
  return rxn_smarts

def insert_atom():
  choices = [['single',['C','N','O','S'],4*[1.0/4.0]],
             ['double',['C','N'],2*[1.0/2.0]],
             ['triple',['C'],[1.0]] ]
  p_BO = [0.60,0.35,0.05]
  
  index = np.random.choice(list(range(3)), p=p_BO)
  
  BO, atom_list, p = choices[index]
  new_atom = np.random.choice(atom_list, p=p)
  
  if BO == 'single': 
    rxn_smarts = '[*:1]~[*:2]>>[*:1]X[*:2]'.replace('X',new_atom)
  if BO == 'double': 
    rxn_smarts = '[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]'.replace('X',new_atom)
  if BO == 'triple': 
    rxn_smarts = '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]'.replace('X',new_atom)
    
  return rxn_smarts

def change_bond_order():
  choices = ['[*:1]!-[*:2]>>[*:1]-[*:2]','[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]',
             '[*:1]#[*:2]>>[*:1]=[*:2]','[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]']
  p = [0.45,0.45,0.05,0.05]
  
  return np.random.choice(choices, p=p)

def delete_cyclic_bond():
  return '[*:1]@[*:2]>>([*:1].[*:2])'

def add_ring():
  choices = ['[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1',
             '[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1',
             '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1',
             '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1'] 
  p = [0.05,0.05,0.45,0.45]
  
  return np.random.choice(choices, p=p)

def change_atom(mol):
  choices = ['#6','#7','#8','#9','#16','#17','#35'] 
  p = [0.15,0.15,0.14,0.14,0.14,0.14,0.14]
  
  X = np.random.choice(choices, p=p)
  while not mol.HasSubstructMatch(Chem.MolFromSmarts('['+X+']')):
    X = np.random.choice(choices, p=p)
  Y = np.random.choice(choices, p=p)
  while Y == X:
    Y = np.random.choice(choices, p=p)
  
  return '[X:1]>>[Y:1]'.replace('X',X).replace('Y',Y)

def aromatic_substitution(mol):
    # I should define the probaility for this reactions according to the PACs compounds table
    # and other papers.
    choices = ['[*;r6:1]~[#6:2]>>[n+;H0;r6:1][C:3][#6:2]',           #Diazonium salt addition
               '[*;!D3;!H0:1]>>[*;!D3:1][NX3+:2]([OX1-:3])=[OX1:4]', #Nitro addition
               '[*;!D3;!H0:1]>>[*;!D3:1]C#N',                        #Ciano addition
               '[*;!D3;!H0:1]>>[*;!D3:1][CX3:2](=[OX1:3])[OX2:4]',   #Carboxilic acid addition
               '[*;!D3;!H0:1]>>[*;!D3:1][CX3:2](=[OX1:3])',          #Aldehide addition 
               '[*;!D3;!H0:1]>>[*;!D3:1][O:2][C:3]',                 #Metoxy addition
               '[*;!D3;!H0:1]>>[*;!D3:1][C:2]([F:3])([F:4])([F:5])'] #CF3 addition
    p = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
    return np.random.choice(choices, p=p)

def mutate(mol,mutation_rate,pos):

  if random.random() > mutation_rate:
    return mol
  
  Chem.Kekulize(mol,clearAromaticFlags=True)
  p = [0.13,0.12,0.12,0.12,0.12,0.13,0.13,0.13]
  for i in range(10):
    rxn_smarts_list = 8*['']
    rxn_smarts_list[0] = insert_atom()
    rxn_smarts_list[1] = change_bond_order()
    rxn_smarts_list[2] = delete_cyclic_bond()
    rxn_smarts_list[3] = add_ring()
    rxn_smarts_list[4] = delete_atom()
    rxn_smarts_list[5] = change_atom(mol)
    rxn_smarts_list[6] = append_atom()
    rxn_smarts_list[7] = aromatic_substitution(mol)
    rxn_smarts = np.random.choice(rxn_smarts_list, p=p) 
    
    #print('mutation',rxn_smarts)
    
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)

    #mol.UpdatePropertyCache()
    #Chem.GetSymmSSSR(mol)
    new_mol_trial = rxn.RunReactants((mol,))
    
    new_mols = []
    for m in new_mol_trial:
      m = m[0]
      #print(Chem.MolToSmiles(m),co.mol_OK(m),co.mol_OK(m),co.linker_OK(m,pos))
      if co.mol_OK(m) and co.ring_OK(m) and co.linker_OK(m,pos):
        new_mols.append(m)
    
    if len(new_mols) > 0:
      return random.choice(new_mols)
  
  return None

if __name__ == "__main__":
    pass
