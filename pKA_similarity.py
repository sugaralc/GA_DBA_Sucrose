'''
Written by Gustavo Lara-Cruz 2024
'''

from rdkit import Chem
from rdkit.Chem import AllChem
import csv

#from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

#Probably I should remove the PBAs with alkyl chains because some of then have a low pKa, 
#like the Wulff PBAs, with a pKa=4.7 but given the similarity with complete alkyl chains 
#in other PBA, assign an incorrect pKa
PBAs_fps = list()
pKas_data = []
with open('PBA_data4sim.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            mol = Chem.MolFromSmiles(row[0])
            pKas_data.append(row[1])
            PBAs_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4086))
            line_count += 1

def dummy2BO3_sim(mol):
    """
    Function to convert the dummy atoms *98 and *99 into boronic groups
    """
    dummy_glc = "[" + str(98) + "*]"
    dummy_frc = "[" + str(99) + "*]"
    dummy_H1 = "[" + str(96) + "*]"
    dummy_H2 = "[" + str(97) + "*]"
    ligBO3 = Chem.ReplaceSubstructs(mol,
                                    Chem.MolFromSmiles(dummy_glc),
                                    Chem.MolFromSmiles('[B-](O)(O)(O)'),
                                    replaceAll=True)

    ligBO3_2 = Chem.ReplaceSubstructs(ligBO3[0],
                                 Chem.MolFromSmiles(dummy_frc),
                                 Chem.MolFromSmiles('[B-](O)(O)(O)'),
                                 replaceAll=True)

    em = Chem.RWMol(ligBO3_2[0])
    em.BeginBatchEdit()
    for atom in em.GetAtoms():
        #print(atom.GetIdx(),atom.GetAtomicNum())
        if atom.GetAtomicNum() == 0:
            #print('Print dummy atom',atom.GetIdx())
            em.RemoveAtom(atom.GetIdx())
    em.CommitBatchEdit()

    return(em)

def pKaFromSimmilarity(smiles):
    idx = 0
    max = -1
    mol = dummy2BO3_sim(Chem.MolFromSmiles(smiles))
    #Chem.SanitizeMol(mol)
    Chem.GetSymmSSSR(mol)
    fp_mol = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=4086)
    for fp in PBAs_fps:
        Tan_sim = Chem.DataStructs.TanimotoSimilarity(fp_mol,fp)
        Cosine_sim = Chem.DataStructs.CosineSimilarity(fp_mol,fp)
        Dice_sim = Chem.DataStructs.DiceSimilarity(fp_mol,fp)
        Consensus = (Tan_sim + Cosine_sim + Dice_sim)/3

        if Consensus > max:
            max = Consensus
            idx_max = idx
        idx += 1
        #print('pka from similarity',idx_max,max)
    return pKas_data[idx_max]


if __name__ == "__main__":
  #smiles = '[97*]C1=CC2=C(C=C3CCC4=C(C([98*])=CN=C4)C3=C2)CC1'
  smiles = '[96*][C@@H]1C=CC2=C(C1)C1=C(C=C2)C2=C(C=C1)C([99*])=CC(N)=C2'

  print(pKaFromSimmilarity(smiles))


