from dataclasses import dataclass, field

from rdkit import Chem

hartree2kcalmol = 627.5094740631


@dataclass
class Individual: # I should add to this class the additional terms for the fitness function. #GALC change 20/02/24
    rdkit_mol: Chem.rdchem.Mol = field(repr=False, compare=False) #Here I will save the structure of the complex  DBA.Suc
    idx: str = field(default=None, compare=True, repr=True)
    smiles: str = field(init=False, compare=True, repr=True)
    score: float = field(default=None, repr=False, compare=False)
    energy: float = field(default=None, repr=False, compare=False)
    sa_score: float = field(default=None, repr=False, compare=False)
    structure: tuple = field(default=None, compare=False, repr=False) #Here I will save the structure of the complex DBA.Suc
    pKa_max: float = field(default=None, repr=False, compare=False) #The higher pKa of the tweezer
    LogP: float = field(default=None, repr=False, compare=False)#Difference between the dot products of the free DBA and the DBA.Suc complex to measure how far is the position of the boronic groups in the free DBA respect the optimal bonding in the DBA.Suc complex.

    def __post_init__(self):
        self.smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(self.rdkit_mol)))


def read_xyz(content, element_symbols=False):
    content = content.strip().split("\n")
    del content[:2]

    atomic_symbols = []
    atomic_positions = []

    for line in content:
        line = line.split()
        atomic_symbols.append(line[0])
        atomic_positions.append(list(map(float, line[1:])))

    if element_symbols:
        pt = Chem.GetPeriodicTable()
        atomic_numbers = list()
        for atom in atomic_symbols:
            atomic_numbers.append(pt.GetAtomicNumber(atom))
        return atomic_numbers, atomic_positions
    else:
        return atomic_symbols, atomic_positions
