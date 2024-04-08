'''
Written by Gustavo Lara-Cruz 2024
'''

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from random import randrange

import random
import numpy as np

from rdkit import rdBase
#rdBase.DisableLog('rdApp.error')
#from rdkit import RDLogger
#RDLogger.DisableLog('rdApp.*')


def Glcyl():
    choices = ['ortho','meta','para']
    p = [0.4,0.4,0.2] 

    ph_subs = np.random.choice(choices, p=p)

    if ph_subs == 'meta':

        PBA_meta=[['C-O-C1=CC=C(-[96*])=CC1-[98*]', 9.0,-1], # 2-Methoxyphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['N-C1=CC(-[96*])=CC(-[98*])=C1', 8.9,-1], # 3-Aminophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C1=CC(-[96*])=CC(-[98*])=C1', 8.8,-1], # Phenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC=C(-[98*])C=C1-[96*]', 8.6,-1], # 4-Fluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051 
        ['ClC1=CC(Cl)=C(-[98*])C=C1-[96*]', 8.5,-1], # 2,4-Diclorophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['ClC1=CC=C(-[98*])C(Cl)=C1-[96*]', 8.5,-1], # 2,4-Diclorophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['Br-C1=CC=C(-[98*])C=C1-[96*]', 8.8,-1], # 4-Bromophenylboronic acid DOI:10.1016/j.tet.2004.08.051 
        ['N-C-C1=CC=C(-[98*])C=C1-[96*]', 8.3,-1], # 4-Aminomethylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CN=CC(-[98*])=C1', 8.1,-1], # 3-Pyridinylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC(-[98*])=CC=N1', 8.0,-1], # 4-Pyridinylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['O-C(=O)-C1=CC=C(-[98*])C=C1-[96*]', 8.0,-1], # 4-Carboxyphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC(-[96*])=CC(-[98*])=C1', 8.0,-1], # 3-Acetophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=C(Cl)C=C(-[98*])C=C1-[96*]', 7.8,-1], # 3-Chloro-4-fluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC(-[98*])=CC(-C=O)=C1', 7.8,-1], # 3-Formylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC=C(-[98*])C=C1-[96*]', 7.7,-1], # 4-Acetylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC(-[98*])=CC=C1-C=O', 7.6,-1], # 4-Formylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(F)=C(-[98*])C=C1-[96*]', 7.6,-1], # 2,4-Difluorophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC=C(-[98*])C(F)=C1-[96*]', 7.6,-1], # 2,4-Difluorophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['[O-]-[N+](=O)-C1=CC(-[96*])=CC(-[98*])=C1', 7.1, -1], # 3-Nitrophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(-[98*])=C(F)C(-[96*])=C1F', 6.8,-1], # 3,4,5-Trifluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=C(F)C(-[96*])=CC(-[98*])=C1F', 6.8,-1], # 2,3,4-Trifluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(-[98*])=C(F)C(-[96*])=C1F', 6.7,-1], # 2,4,5-Trifluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C-[N+](-C)-C-C1=CC=C(-[96*])C=C1-[98*]', 5.3,-1], # 2-Dimethylaminomethylphenylboronic acid (DAPBA) DOI:10.1016/j.tet.2004.08.051
        ['[O-]-[N+](=O)-C1=CC(-[96*])=C(F)C(-[98*])=C1', 6.0,-1], # 2-Fluoro-5-nitrophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C-C1=CC(-[98*])=C[N+](-[#6]-[96*])=C1', 4.4,-1], # 5-Methylpyridine-3-boronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C-[N+]1=CC=CC(-[98*])=C1', 4.4,-1], # 3-Methylpyridineboronic acid  DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC=C(-C=O)C(-[98*])=C1', 7.31,-1], # 2-Formylphenilboronic acid DOI:10.3390/molecules25040799
        ['FC1=CC(-[96*])=CC(-[98*])=C1', 7.5,-1], # 3-Fluorophenilboronic acid DOI:10.3390/molecules25040799
        ['FC(F)(F)C1=CC(-[96*])=CC(-[98*])=C1', 7.85,-1], # 3-Metiltrifluorophenilboronic acid DOI:10.3390/molecules25040799       
        ['[96*]-C1=CC(-[98*])=CC=C1S(=O)(=O)C-C-C=C', 7.1,-1], # 4-(3-butenylsulfonyl)phenylboronic acid BSPBA DOI:10.1016/j.ab.2007.09.001 Bonding to silicon trough double bond in alkyl chain
        ['[96*]-C1=CC(-[98*])=CC=C1S(=O)(=O)N-C-C=C', 7.4 ,-1], # 4-(N-allylsulfamoyl)phenylboronic acid BSPBA DOI:10.1016/j.ab.2007.09.001 Bonding to silicon trough double bond in alkyl chain
        ['N-C1=CC(-[98*])=CC(-[96*])=C1-[N+](-[O-])=O', 7.1,-1], # (3-amino-4nitrophenyl)boronic acid DOI:10.3390/molecules25040799 Binding to silica the amino group
        ['[96*]-C1=CC(-[98*])=CC(-N-C(=O)-C=C)=C1', 8.2,-1], # 3-acrylamidophenylboronic acid (AAPBA) See the review DOI: 10.1039/C5CS00013K
        ['[96*]-C1=CC(-[98*])=CC=C1-C(=O)-N-C-C-N-C(=O)-C=C', 7.8,-1], # 4-(1,6-dioxo-2,5-diaza-7-oxamyl-)-phenylboronic acid (DDOPBA) DOI:10.1039/B920319B Bonding to silicon trough double bond in alkyl chain See the review DOI: 10.1039/C5CS00013K
        ['FC1=C(-[96*])C=C(-[98*])C(F)=C1-C=O', 6.5,-1], # 2,4-difluoro-3-formyl-phenylboronicacid (DFFPBA) See the review DOI: 10.1039/C5CS00013K
        #['[96*]-c1ccc2-[#6]-[#8]-[98*]-c2c1', 7.2,0], # Benzoboroxole DOI: 10.1021/jo800788s Bonding with trans 4-6 diols prefered over trans 3-4 diol as in glucose.
        ['[96*]-C-1=C-C(-[98*])=C-S-1', 8.1 ,-1], # 3-thiopeneboronic acid DOI: 10.3390/chemosensors10070251 Read this paper, has some thermochemical measures
        ['FC1=NC=C(-[96*])C=C1-[98*]', 7.1,-1], # 2-fluoro-3-pyridylboronic acid or 2F-3-PyBA DOI: 10.1021/ol5036003
        ['FC1=NC=C(-[98*])C=C1-[96*]', 7.0,-1], # 2-fluoro-5-pyridylboronic acid or 2F-5-PyBA  DOI: 10.1021/ol5036003
        ['[96*]-C-[N+]1=CN=CC(-[98*])=C1', 6.2,-1], # pyrimidine-5-boronic acid DOI:10.1039/c7sc01905j
        ['[96*]-C1=CN=CC(-[98*])=C1', 4.4, -1], # 3-pyridylboronic acid DOI:10.1039/c7sc01905j 
        ['O-C(=O)-C1=NC=C(-[98*])C=C1-[96*]', 4.2, -1], # 5-boronopicolinic acid DOI:10.1039/c7sc01905j 
        ['O-C(=O)-C1=CC=C(-[98*])C=[N+]1-C-[96*]', 4.2, -1], # 5-boronopicolinic acid DOI:10.1039/c7sc01905j 
        ['C-C-C-N-C(=O)-C1=NC=C(-[98*])C=C1-[96*]', 4.2, -1], # (6-propylcarbamoyl)pyridine-3-boronic acid DOI:10.1039/c7sc01905j
        ['C-C-C-N-C(=O)-C1=CC=C(-[98*])C=[N+]1-C-[96*]', 4.2, -1], # (6-propylcarbamoyl)pyridine-3-boronic acid DOI:10.1039/c7sc01905j
        ['O-C(=O)-C-[N+]1=CC(-[96*])=CC(-[98*])=C1', 4.4,-1], # 3-borono-1-(carboxymethyl)pyridine DOI:10.1039/c7sc01905j
        ['C-C-C(=O)-N-C1=CC(-[96*])=CC(-[98*])=C1', 8.3, -1], # 3-propionamidophenylboronic acid DOI:10.1039/c7sc01905j
        ['CS(=O)(=O)C1=CC=C(-[98*])C=C1-[96*]', 7.1,-1], # 4-(methylsulfonyl)benzeneboronic acid DOI:10.1039/c7sc01905j pKa approx   
        ['FC1=NC=C(-[96*])C=C1-[98*]', 6.3,-1], # 2-Fluoro-3-pyridyl boronic acid DOI:10.1039/c7sc01905j
        ['FC1=C(-[98*])C=CC=[N+]1-C-[96*]', 6.3,-1]] # 2-Fluoro-3-pyridyl boronic acid DOI:10.1039/c7sc01905j
        
        p = len(PBA_meta)*[1/len(PBA_meta)]
        index = np.random.choice(list(range(len(PBA_meta))), p = p)
        PBA_mol, pKa, dummy = PBA_meta[index]


        return Chem.MolFromSmiles(PBA_mol), str(pKa)

    if ph_subs == 'ortho':

        PBA_ortho=[['C-O-C1=CC=C(-[96*])C(-[98*])=C1', 9.0,-1], # 2-Methoxyphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['N-C1=CC=C(-[96*])C(-[98*])=C1', 8.9,-1], # 3-Aminophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['N-C1=CC=CC(-[98*])=C1-[96*]', 8.9,-1], # 3-Aminophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC=CC=C1-[98*]', 8.8,-1], # Phenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC=C(-[98*])C(-[96*])=C1', 8.6,-1],  # 4-Fluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['ClC1=CC(Cl)=C(-[98*])C(-[96*])=C1', 8.5,-1], # 2,4-Diclorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['BrC1=CC=C(-[98*])C(-[96*])=C1', 8.8,-1], # 4-Bromophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['N-C-C1=CC=C(-[98*])C(-[96*])=C1', 8.3,-1], # 4-Aminomethylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC=NC=C1-[98*]', 8.1,-1], # 3-Pyridinylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CN=CC=C1-[98*]', 8.0,-1], # 4-Pyridinylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['O-C(=O)-C1=CC=C(-[98*])C(-[96*])=C1', 8.0, -1], # 4-Carboxyphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC=CC(-[98*])=C1-[96*]', 8.0,-1], # 3-Acetophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC=C(-[96*])C(-[98*])=C1', 8.0,-1], # 3-Acetophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(-[96*])=C(-[98*])C=C1Cl', 7.8,-1], # 3-Chloro-4-fluorophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC=C(-[98*])C(-[96*])=C1Cl', 7.8,-1], # 3-Chloro-4-fluorophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC=C(-[98*])C(-[96*])=C1', 7.7, -1], # 4-Acetylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC(-C=O)=CC=C1-[98*]', 8.0, -1], # 3-Acetophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(-[96*])=C(-[98*])C=C1Cl', 7.8,-1], # 3-Chloro-4-fluorophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC=C(-[98*])C(-[96*])=C1Cl', 7.8,-1], # 3-Chloro-4-fluorophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=C(-[98*])C=CC=C1-C=O', 7.8,-1], # 3-Formylphenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC=C(-C=O)C=C1-[98*]', 7.8,-1], # 3-Formylphenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC=C(-[98*])C(-[96*])=C1', 7.7,-1], # 4-Acetylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C(=O)-C1=CC=C(-[98*])C(-[96*])=C1', 7.6,-1], # 4-Formylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(F)=C(-[98*])C(-[96*])=C1', 7.6, -1], # 2,4-Difluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[O-]-[N+](=O)-C1=CC=C(-[96*])C(-[98*])=C1', 7.1,-1], # 3-Nitrophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['[O-]-[N+](=O)-C1=CC=CC(-[98*])=C1-[96*]', 7.1,-1], # 3-Nitrophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC=C(F)C(-[98*])=C1-[96*]', 7.0,-1], # 2,5-Diflourophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(F)=C(-[98*])C(-[96*])=C1F', 6.8,-1], # 3,4,5-Trifluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(-[96*])=C(-[98*])C(F)=C1F', 6.8,-1], # 2,3,4-Trifluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051                          
        ['FC1=CC=C(-C=O)C(-[98*])=C1-[96*]', 6.72,-1], # 5-ﬂuoro-2-formylphenylboronic Acid DOI:10.3390/molecules25040799
        ['FC(F)(F)C1=CC=C(-C=O)C(-[98*])=C1-[96*]', 6.72,-1], #5-Triﬂuoromethyl-2-formylphenylboronic Acid DOI:10.3390/molecules25040799
        ['[O-]-[N+](=O)-C1=CC=C(F)C(-[98*])=C1-[96*]', 6.0,-1], # 2-Fluoro-5-nitrophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC=CC(-C=O)=C1-[98*]', 7.31,-1], # 2-Formylphenilboronic acid DOI:10.3390/molecules25040799
        ['FC1=CC=C(-[96*])C(-[98*])=C1', 7.5, -1], # 3-Fluorophenilboronic acid DOI:10.3390/molecules25040799
        ['FC1=CC=CC(-[98*])=C1-[96*]', 7.5,-1], # 3-Fluorophenilboronic acid DOI:10.3390/molecules25040799
        ['FC(F)(F)C1=CC=C(-[96*])C(-[98*])=C1', 7.85,-1], # 3-Metiltrifluorophenilboronic acid DOI:10.3390/molecules25040799
        ['[96*]-C1=CC(=CC=C1-[98*])S(=O)(=O)C-C-C=C', 7.1,-1], # 4-(3-butenylsulfonyl)phenylboronic acid BSPBA DOI:10.1016/j.ab.2007.09.001 Bonding to silicon trough double bond in alkyl chain
        ['[96*]-C1=CC(=CC=C1-[98*])S(=O)(=O)N-C-C=C', 7.4,-1], # 4-(N-allylsulfamoyl)phenylboronic acid BSPBA DOI:10.1016/j.ab.2007.09.001 Bonding to silicon trough double bond in alkyl chain
        ['N-C1=CC(-[98*])=C(-[96*])C=C1-[N+](-[O-])=O', 7.1,-1], # (3-amino-4nitrophenyl)boronic acid DOI: 10.1016/S1872-2040(07)60007-3 Binding to silica the amino group
        ['N-C1=C(-[96*])C(-[98*])=CC=C1-[N+](-[O-])=O', 7.1,-1], # (3-amino-4nitrophenyl)boronic acid DOI: 10.1016/S1872-2040(07)60007-3 Binding to silica the amino group
        ['[96*]-C1=C(-[98*])C=CC=C1-N-C(=O)-C=C', 8.2,-1], # 3-acrylamidophenylboronic acid (AAPBA) See the review DOI: 10.1039/C5CS00013K
        ['[96*]-C1=CC=C(-N-C(=O)-C=C)C=C1-[98*]', 8.2,-1], # 3-acrylamidophenylboronic acid (AAPBA) See the review DOI: 10.1039/C5CS00013K
        ['[96*]-C1=CC(=CC=C1-[98*])-C(=O)-N-C-C-N-C(=O)-C=C', 7.8 ,-1], # 4-(1,6-dioxo-2,5-diaza-7-oxamyl-)-phenylboronic acid (DDOPPA)  See the review DOI: 10.1039/C5CS00013K
        ['FC1=CC(-[96*])=C(-[98*])C(F)=C1-C=O', 6.5,-1], # 2,4-difluoro-3-formyl-phenylboronicacid (DFFPBA) See the review DOI: 10.1039/C5CS00013K
        #['[96*]-c1cccc2-[#6]-[#8]-[98*]-c12', 7.2,0], # Benzoboroxole DOI: 10.1021/jo800788s Bonding with trans 4-6 diols prefered over trans 3-4 diol as in glucose.
        #['[#8]-[#6](=O)-c1ccc2-[#6]-[#8]-[98*]-c2c1-[96*]', 7.2,0], # Benzoboroxolebenzaldehide Binding to polimer throug carboxilic acid. Asummed to be similar to Benzoboroxole DOI: 10.1021/jo800788s
        ['[96*]-C1=C-S-C=C1-[98*]', 8.1,-1], # 3-thiopeneboronic acid DOI: 10.3390/chemosensors10070251 Read this paper, has some thermochemical measures
        ['[96*]-C1=C(-[98*])-C=C-S1', 8.1,-1], # 3-thiopeneboronic acid DOI: 10.3390/chemosensors10070251 Read this paper, has some thermochemical measures
        ['FC1=NC=CC(-[96*])=C1-[98*]', 7.1,-1], # 2-fluoro-3-pyridylboronic acid or 2F-3-PyBA   DOI: 10.1021/ol5036003
        ['FC1=CC(-[96*])=C(-[98*])C=N1', 7.0,-1], # 2-fluoro-5-pyridylboronic acid or 2F-5-PyBA  DOI: 10.1021/ol5036003
        ['FC1=CC=C(-[98*])C(-[96*])=N1', 7.0,-1], # 2-fluoro-5-pyridylboronic acid or 2F-5-PyBA DOI: 10.1021/ol5036003
        ['[96*]-C1=NC=NC=C1-[98*]', 6.2, -1], # pyrimidine-5-boronic acid DOI:10.1039/c7sc01905j
        ['[96*]-C1=CC=NC=C1-[98*]',  4.4, -1],  # 3-pyridylboronic acid DOI:10.1039/c7sc01905j
        ['[96*]-C1=NC=CC=C1-[98*]',  4.4, -1],  # 3-pyridylboronic acid DOI:10.1039/c7sc01905j
        ['O-C(=O)-C1=CC(-[96*])=C(-[98*])C=N1', 4.2, -1], # 5-boronopicolinic acid DOI:10.1039/c7sc01905j
        ['O-C(=O)-C1=CC=C(-[98*])C(-[96*])=N1', 4.2, -1], # 5-boronopicolinic acid DOI:10.1039/c7sc01905j
        ['C-C-C-N-C(=O)-C1=CC(-[96*])=C(-[98*])C=N1', 4.2,-1], # (6-propylcarbamoyl)pyridine-3-boronic acid DOI:10.1039/c7sc01905j
        ['C-C-C-N-C(=O)-C1=CC=C(-[98*])C(-[96*])=N1', 4.2,-1], # (6-propylcarbamoyl)pyridine-3-boronic acid DOI:10.1039/c7sc01905j
        ['O-C(=O)-C-[N+]1=CC=CC(-[98*])=C1-[96*]', 4.4,-1], # 3-borono-1-(carboxymethyl)pyridine DOI:10.1039/c7sc01905j
        ['O-C(=O)-C-[N+]1=CC=C(-[96*])C(-[98*])=C1', 4.4,-1], # 3-borono-1-(carboxymethyl)pyridine DOI:10.1039/c7sc01905j
        ['C-C-C(=O)-N-C1=CC=C(-[96*])C(-[98*])=C1', 8.3,-1], # 3-propionamidophenylboronic acid DOI:10.1039/c7sc01905j
        ['C-C-C(=O)-N-C1=CC=CC(-[98*])=C1-[96*]', 8.3,-1], # 3-propionamidophenylboronic acid DOI:10.1039/c7sc01905j
        ['CS(=O)(=O)C1=CC=C(-[98*])C=C1-[96*]', 7.1,-1], # 4-(methylsulfonyl)benzeneboronic acid DOI:10.1039/c7sc01905j pKa approx
        ['FC1=NC=CC(-[96*])=C1-[98*]', 6.3,-1]] # 2-Fluoro-3-pyridyl boronic acid DOI:10.1039/c7sc01905j

        p = len(PBA_ortho)*[1/len(PBA_ortho)]
        index = np.random.choice(list(range(len(PBA_ortho))), p = p)
        PBA_mol, pKa, dummy = PBA_ortho[index]

        return Chem.MolFromSmiles(PBA_mol), str(pKa)

    if ph_subs == 'para':
        
        PBA_para=[['N-C1=CC(-[98*])=CC=C1-[96*]', 8.9,-1], # 3-Aminophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC=C(-[98*])C=C1', 8.8,-1], # Phenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC=C(-[98*])C=N1', 8.1,-1], # 3-Pyridinylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC(-[98*])=CC=C1-[96*]', 8.0,-1], # 3-Acetophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC=C(-[98*])C=C1-C=O', 7.8,-1], # 3-Formylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[O-]-[N+](=O)-C1=CC(-[98*])=CC=C1-[96*]', 7.1,-1], # 3-Nitrophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(-[98*])=C(F)C=C1-[96*]', 7.0,-1], # 2,5-Diflourophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[O-]-[N+](=O)-C1=CC(-[98*])=C(F)C=C1-[96*]', 6.0,-1], # 2-Fluoro-5-nitrophenylboronic acid DOI:10.1016/j.tet.2004.08.051        
        ['[96*]-C-[N+]1=CC=C(-[98*])C=C1', 4.4,-1], # 4-Methylpyridineboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[96*]-C1=CC=C(-[98*])C(-C=O)=C1', 7.31,-1], # 2-Formylphenilboronic acid DOI:10.3390/molecules25040799
        ['FC1=CC(-[98*])=CC=C1-[96*]', 7.5,-1], # 3-Fluorophenilboronic acid DOI:10.3390/molecules25040799
        ['FC(F)(F)C1=CC(-[98*])=CC=C1-[96*]', 7.85,-1], # 3-Metiltrifluorophenilboronic acid DOI:10.3390/molecules25040799
        ['[96*]-C1=CC=C(-[98*])C=C1-N-C(=O)-C=C', 8.2 ,-1], # 3-acrylamidophenylboronic acid (AAPBA) See the review DOI: 10.1039/C5CS00013K 
        ['[96*]-C1=CC=C(-[98*])C=C1-C(=O)-N-C-C-N-C(=O)-C=C',7.8,-1], # 4-(1,6-dioxo-2,5-diaza-7-oxamyl-)-phenylboronic acid (DDOPPA)
        #['[96*]-c1ccc2-[98*]-[#8]-[#6]-c2c1'. 7.2,0], # Benzoboroxole DOI: 10.1021/jo800788s Bonding with trans 4-6 diols prefered over trans 3-4 diol as in glucose.
        #['[#8]-[#6](=O)-c1cc2-[98*]-[#8]-[#6]-c2cc1-[96*]', 7.2,0], # Benzoboroxolebenzaldehide Binding to polimer throug carboxilic acid. Asummed to be similar to Benzoboroxole DOI: 10.1021/jo800788s
        ['FC1=NC(-[96*])=CC=C1-[98*]', 7.1,-1], # 2-fluoro-3-pyridylboronic acid or 2F-3-PyBA DOI: 10.1021/ol5036003
        ['[96*]-C1=NC=C(-[98*])C=N1', 6.2,-1], # pyrimidine-5-boronic acid DOI:10.1039/c7sc01905j
        ['[96*]-C1=CC=C(-[98*])C=N1', 4.4,-1],  # 3-pyridylboronic acid DOI:10.1039/c7sc01905j
        ['O-C(=O)-C-[N+]1=CC(-[98*])=CC=C1-[96*]', 4.4,-1], # 3-borono-1-(carboxymethyl)pyridine DOI:10.1039/c7sc01905j
        ['C-C-C(=O)-N-C1=CC(-[98*])=CC=C1-[96*]', 8.3, -1], # 3-propionamidophenylboronic acid DOI:10.1039/c7sc01905j
        ['FC1=NC(-[96*])=CC=C1-[98*]',6.3,-1]] # 2-Fluoro-3-pyridyl boronic acid DOI:10.1039/c7sc01905j

        p = len(PBA_para)*[1/len(PBA_para)]
        index = np.random.choice(list(range(len(PBA_para))), p = p)
        PBA_mol, pKa, dummy = PBA_para[index]

        return Chem.MolFromSmiles(PBA_mol), str(pKa)

def Frcyl():
    choices = ['ortho','meta','para']
    p = [0.4,0.4,0.2] 

    ph_subs = np.random.choice(choices, p=p)

    if ph_subs == 'meta':

        PBA_meta=[['C-O-C1=CC=C(-[97*])=CC1-[99*]', 9.0,-1], # 2-Methoxyphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['N-C1=CC(-[97*])=CC(-[99*])=C1', 8.9,-1], # 3-Aminophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C1=CC(-[97*])=CC(-[99*])=C1', 8.8,-1], # Phenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC=C(-[99*])C=C1-[97*]', 8.6,-1], # 4-Fluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051 
        ['ClC1=CC(Cl)=C(-[99*])C=C1-[97*]', 8.5,-1], # 2,4-Diclorophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['ClC1=CC=C(-[99*])C(Cl)=C1-[97*]', 8.5,-1], # 2,4-Diclorophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['Br-C1=CC=C(-[99*])C=C1-[97*]', 8.8,-1], # 4-Bromophenylboronic acid DOI:10.1016/j.tet.2004.08.051 
        ['N-C-C1=CC=C(-[99*])C=C1-[97*]', 8.3,-1], # 4-Aminomethylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CN=CC(-[99*])=C1', 8.1,-1], # 3-Pyridinylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC(-[99*])=CC=N1', 8.0,-1], # 4-Pyridinylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['O-C(=O)-C1=CC=C(-[99*])C=C1-[97*]', 8.0,-1], # 4-Carboxyphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC(-[97*])=CC(-[99*])=C1', 8.0,-1], # 3-Acetophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=C(Cl)C=C(-[99*])C=C1-[97*]', 7.8,-1], # 3-Chloro-4-fluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC(-[99*])=CC(-C=O)=C1', 7.8,-1], # 3-Formylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC=C(-[99*])C=C1-[97*]', 7.7,-1], # 4-Acetylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC(-[99*])=CC=C1-C=O', 7.6,-1], # 4-Formylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(F)=C(-[99*])C=C1-[97*]', 7.6,-1], # 2,4-Difluorophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC=C(-[99*])C(F)=C1-[97*]', 7.6,-1], # 2,4-Difluorophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['[O-]-[N+](=O)-C1=CC(-[97*])=CC(-[99*])=C1', 7.1, -1], # 3-Nitrophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(-[99*])=C(F)C(-[97*])=C1F', 6.8,-1], # 3,4,5-Trifluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=C(F)C(-[97*])=CC(-[99*])=C1F', 6.8,-1], # 2,3,4-Trifluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(-[99*])=C(F)C(-[97*])=C1F', 6.7,-1], # 2,4,5-Trifluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C-[N+](-C)-C-C1=CC=C(-[97*])C=C1-[99*]', 5.3,-1], # 2-Dimethylaminomethylphenylboronic acid (DAPBA) DOI:10.1016/j.tet.2004.08.051
        ['[O-]-[N+](=O)-C1=CC(-[97*])=C(F)C(-[99*])=C1', 6.0,-1], # 2-Fluoro-5-nitrophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C-C1=CC(-[99*])=C[N+](-[#6]-[97*])=C1', 4.4,-1], # 5-Methylpyridine-3-boronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C-[N+]1=CC=CC(-[99*])=C1', 4.4,-1], # 3-Methylpyridineboronic acid  DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC=C(-C=O)C(-[99*])=C1', 7.31,-1], # 2-Formylphenilboronic acid DOI:10.3390/molecules25040799
        ['FC1=CC(-[97*])=CC(-[99*])=C1', 7.5,-1], # 3-Fluorophenilboronic acid DOI:10.3390/molecules25040799
        ['FC(F)(F)C1=CC(-[97*])=CC(-[99*])=C1', 7.85,-1], # 3-Metiltrifluorophenilboronic acid DOI:10.3390/molecules25040799       
        ['[97*]-C1=CC(-[99*])=CC=C1S(=O)(=O)C-C-C=C', 7.1,-1], # 4-(3-butenylsulfonyl)phenylboronic acid BSPBA DOI:10.1016/j.ab.2007.09.001 Bonding to silicon trough double bond in alkyl chain
        ['[97*]-C1=CC(-[99*])=CC=C1S(=O)(=O)N-C-C=C', 7.4 ,-1], # 4-(N-allylsulfamoyl)phenylboronic acid BSPBA DOI:10.1016/j.ab.2007.09.001 Bonding to silicon trough double bond in alkyl chain
        ['N-C1=CC(-[99*])=CC(-[97*])=C1-[N+](-[O-])=O', 7.1,-1], # (3-amino-4nitrophenyl)boronic acid DOI:10.3390/molecules25040799 Binding to silica the amino group
        ['[97*]-C1=CC(-[99*])=CC(-N-C(=O)-C=C)=C1', 8.2,-1], # 3-acrylamidophenylboronic acid (AAPBA) See the review DOI: 10.1039/C5CS00013K
        ['[97*]-C1=CC(-[99*])=CC=C1-C(=O)-N-C-C-N-C(=O)-C=C', 7.8,-1], # 4-(1,6-dioxo-2,5-diaza-7-oxamyl-)-phenylboronic acid (DDOPBA) DOI:10.1039/B920319B Bonding to silicon trough double bond in alkyl chain See the review DOI: 10.1039/C5CS00013K
        ['FC1=C(-[97*])C=C(-[99*])C(F)=C1-C=O', 6.5,-1], # 2,4-difluoro-3-formyl-phenylboronicacid (DFFPBA) See the review DOI: 10.1039/C5CS00013K
        #['[97*]-c1ccc2-[#6]-[#8]-[99*]-c2c1', 7.2,0], # Benzoboroxole DOI: 10.1021/jo800788s Bonding with trans 4-6 diols prefered over trans 3-4 diol as in glucose.
        ['[97*]-C-1=C-C(-[99*])=C-S-1', 8.1 ,-1], # 3-thiopeneboronic acid DOI: 10.3390/chemosensors10070251 Read this paper, has some thermochemical measures
        ['FC1=NC=C(-[97*])C=C1-[99*]', 7.1,-1], # 2-fluoro-3-pyridylboronic acid or 2F-3-PyBA DOI: 10.1021/ol5036003
        ['FC1=NC=C(-[99*])C=C1-[97*]', 7.0,-1], # 2-fluoro-5-pyridylboronic acid or 2F-5-PyBA  DOI: 10.1021/ol5036003
        ['[97*]-C-[N+]1=CN=CC(-[99*])=C1', 6.2,-1], # pyrimidine-5-boronic acid DOI:10.1039/c7sc01905j
        ['[97*]-C1=CN=CC(-[99*])=C1', 4.4, -1], # 3-pyridylboronic acid DOI:10.1039/c7sc01905j 
        ['O-C(=O)-C1=NC=C(-[99*])C=C1-[97*]', 4.2, -1], # 5-boronopicolinic acid DOI:10.1039/c7sc01905j 
        ['O-C(=O)-C1=CC=C(-[99*])C=[N+]1-C-[97*]', 4.2, -1], # 5-boronopicolinic acid DOI:10.1039/c7sc01905j 
        ['C-C-C-N-C(=O)-C1=NC=C(-[99*])C=C1-[97*]', 4.2, -1], # (6-propylcarbamoyl)pyridine-3-boronic acid DOI:10.1039/c7sc01905j
        ['C-C-C-N-C(=O)-C1=CC=C(-[99*])C=[N+]1-C-[97*]', 4.2, -1], # (6-propylcarbamoyl)pyridine-3-boronic acid DOI:10.1039/c7sc01905j
        ['O-C(=O)-C-[N+]1=CC(-[97*])=CC(-[99*])=C1', 4.4,-1], # 3-borono-1-(carboxymethyl)pyridine DOI:10.1039/c7sc01905j
        ['C-C-C(=O)-N-C1=CC(-[97*])=CC(-[99*])=C1', 8.3, -1], # 3-propionamidophenylboronic acid DOI:10.1039/c7sc01905j
        ['CS(=O)(=O)C1=CC=C(-[99*])C=C1-[97*]', 7.1,-1], # 4-(methylsulfonyl)benzeneboronic acid DOI:10.1039/c7sc01905j pKa approx   
        ['FC1=NC=C(-[97*])C=C1-[99*]', 6.3,-1], # 2-Fluoro-3-pyridyl boronic acid DOI:10.1039/c7sc01905j
        ['FC1=C(-[99*])C=CC=[N+]1-C-[97*]', 6.3,-1]] # 2-Fluoro-3-pyridyl boronic acid DOI:10.1039/c7sc01905j
        
        p = len(PBA_meta)*[1/len(PBA_meta)]
        index = np.random.choice(list(range(len(PBA_meta))), p = p)
        PBA_mol, pKa, dummy = PBA_meta[index]


        return Chem.MolFromSmiles(PBA_mol), str(pKa)

    if ph_subs == 'ortho':

        PBA_ortho=[['C-O-C1=CC=C(-[97*])C(-[99*])=C1', 9.0,-1], # 2-Methoxyphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['N-C1=CC=C(-[97*])C(-[99*])=C1', 8.9,-1], # 3-Aminophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['N-C1=CC=CC(-[99*])=C1-[97*]', 8.9,-1], # 3-Aminophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC=CC=C1-[99*]', 8.8,-1], # Phenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC=C(-[99*])C(-[97*])=C1', 8.6,-1],  # 4-Fluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['ClC1=CC(Cl)=C(-[99*])C(-[97*])=C1', 8.5,-1], # 2,4-Diclorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['BrC1=CC=C(-[99*])C(-[97*])=C1', 8.8,-1], # 4-Bromophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['N-C-C1=CC=C(-[99*])C(-[97*])=C1', 8.3,-1], # 4-Aminomethylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC=NC=C1-[99*]', 8.1,-1], # 3-Pyridinylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CN=CC=C1-[99*]', 8.0,-1], # 4-Pyridinylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['O-C(=O)-C1=CC=C(-[99*])C(-[97*])=C1', 8.0, -1], # 4-Carboxyphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC=CC(-[99*])=C1-[97*]', 8.0,-1], # 3-Acetophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC=C(-[97*])C(-[99*])=C1', 8.0,-1], # 3-Acetophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(-[97*])=C(-[99*])C=C1Cl', 7.8,-1], # 3-Chloro-4-fluorophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC=C(-[99*])C(-[97*])=C1Cl', 7.8,-1], # 3-Chloro-4-fluorophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC=C(-[99*])C(-[97*])=C1', 7.7, -1], # 4-Acetylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC(-C=O)=CC=C1-[99*]', 8.0, -1], # 3-Acetophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(-[97*])=C(-[99*])C=C1Cl', 7.8,-1], # 3-Chloro-4-fluorophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC=C(-[99*])C(-[97*])=C1Cl', 7.8,-1], # 3-Chloro-4-fluorophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=C(-[99*])C=CC=C1-C=O', 7.8,-1], # 3-Formylphenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC=C(-C=O)C=C1-[99*]', 7.8,-1], # 3-Formylphenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC=C(-[99*])C(-[97*])=C1', 7.7,-1], # 4-Acetylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C(=O)-C1=CC=C(-[99*])C(-[97*])=C1', 7.6,-1], # 4-Formylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(F)=C(-[99*])C(-[97*])=C1', 7.6, -1], # 2,4-Difluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[O-]-[N+](=O)-C1=CC=C(-[97*])C(-[99*])=C1', 7.1,-1], # 3-Nitrophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['[O-]-[N+](=O)-C1=CC=CC(-[99*])=C1-[97*]', 7.1,-1], # 3-Nitrophenylboronic acid 2 DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC=C(F)C(-[99*])=C1-[97*]', 7.0,-1], # 2,5-Diflourophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(F)=C(-[99*])C(-[97*])=C1F', 6.8,-1], # 3,4,5-Trifluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(-[97*])=C(-[99*])C(F)=C1F', 6.8,-1], # 2,3,4-Trifluorophenylboronic acid DOI:10.1016/j.tet.2004.08.051                          
        ['FC1=CC=C(-C=O)C(-[99*])=C1-[97*]', 6.72,-1], # 5-ﬂuoro-2-formylphenylboronic Acid DOI:10.3390/molecules25040799
        ['FC(F)(F)C1=CC=C(-C=O)C(-[99*])=C1-[97*]', 6.72,-1], #5-Triﬂuoromethyl-2-formylphenylboronic Acid DOI:10.3390/molecules25040799
        ['[O-]-[N+](=O)-C1=CC=C(F)C(-[99*])=C1-[97*]', 6.0,-1], # 2-Fluoro-5-nitrophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC=CC(-C=O)=C1-[99*]', 7.31,-1], # 2-Formylphenilboronic acid DOI:10.3390/molecules25040799
        ['FC1=CC=C(-[97*])C(-[99*])=C1', 7.5, -1], # 3-Fluorophenilboronic acid DOI:10.3390/molecules25040799
        ['FC1=CC=CC(-[99*])=C1-[97*]', 7.5,-1], # 3-Fluorophenilboronic acid DOI:10.3390/molecules25040799
        ['FC(F)(F)C1=CC=C(-[97*])C(-[99*])=C1', 7.85,-1], # 3-Metiltrifluorophenilboronic acid DOI:10.3390/molecules25040799
        ['[97*]-C1=CC(=CC=C1-[99*])S(=O)(=O)C-C-C=C', 7.1,-1], # 4-(3-butenylsulfonyl)phenylboronic acid BSPBA DOI:10.1016/j.ab.2007.09.001 Bonding to silicon trough double bond in alkyl chain
        ['[97*]-C1=CC(=CC=C1-[99*])S(=O)(=O)N-C-C=C', 7.4,-1], # 4-(N-allylsulfamoyl)phenylboronic acid BSPBA DOI:10.1016/j.ab.2007.09.001 Bonding to silicon trough double bond in alkyl chain
        ['N-C1=CC(-[99*])=C(-[97*])C=C1-[N+](-[O-])=O', 7.1,-1], # (3-amino-4nitrophenyl)boronic acid DOI: 10.1016/S1872-2040(07)60007-3 Binding to silica the amino group
        ['N-C1=C(-[97*])C(-[99*])=CC=C1-[N+](-[O-])=O', 7.1,-1], # (3-amino-4nitrophenyl)boronic acid DOI: 10.1016/S1872-2040(07)60007-3 Binding to silica the amino group
        ['[97*]-C1=C(-[99*])C=CC=C1-N-C(=O)-C=C', 8.2,-1], # 3-acrylamidophenylboronic acid (AAPBA) See the review DOI: 10.1039/C5CS00013K
        ['[97*]-C1=CC=C(-N-C(=O)-C=C)C=C1-[99*]', 8.2,-1], # 3-acrylamidophenylboronic acid (AAPBA) See the review DOI: 10.1039/C5CS00013K
        ['[97*]-C1=CC(=CC=C1-[99*])-C(=O)-N-C-C-N-C(=O)-C=C', 7.8 ,-1], # 4-(1,6-dioxo-2,5-diaza-7-oxamyl-)-phenylboronic acid (DDOPPA)  See the review DOI: 10.1039/C5CS00013K
        ['FC1=CC(-[97*])=C(-[99*])C(F)=C1-C=O', 6.5,-1], # 2,4-difluoro-3-formyl-phenylboronicacid (DFFPBA) See the review DOI: 10.1039/C5CS00013K
        #['[97*]-c1cccc2-[#6]-[#8]-[99*]-c12', 7.2,0], # Benzoboroxole DOI: 10.1021/jo800788s Bonding with trans 4-6 diols prefered over trans 3-4 diol as in glucose.
        #['[#8]-[#6](=O)-c1ccc2-[#6]-[#8]-[99*]-c2c1-[97*]', 7.2,0], # Benzoboroxolebenzaldehide Binding to polimer throug carboxilic acid. Asummed to be similar to Benzoboroxole DOI: 10.1021/jo800788s
        ['[97*]-C1=C-S-C=C1-[99*]', 8.1,-1], # 3-thiopeneboronic acid DOI: 10.3390/chemosensors10070251 Read this paper, has some thermochemical measures
        ['[97*]-C1=C(-[99*])-C=C-S1', 8.1,-1], # 3-thiopeneboronic acid DOI: 10.3390/chemosensors10070251 Read this paper, has some thermochemical measures
        ['FC1=NC=CC(-[97*])=C1-[99*]', 7.1,-1], # 2-fluoro-3-pyridylboronic acid or 2F-3-PyBA   DOI: 10.1021/ol5036003
        ['FC1=CC(-[97*])=C(-[99*])C=N1', 7.0,-1], # 2-fluoro-5-pyridylboronic acid or 2F-5-PyBA  DOI: 10.1021/ol5036003
        ['FC1=CC=C(-[99*])C(-[97*])=N1', 7.0,-1], # 2-fluoro-5-pyridylboronic acid or 2F-5-PyBA DOI: 10.1021/ol5036003
        ['[97*]-C1=NC=NC=C1-[99*]', 6.2, -1], # pyrimidine-5-boronic acid DOI:10.1039/c7sc01905j
        ['[97*]-C1=CC=NC=C1-[99*]',  4.4, -1],  # 3-pyridylboronic acid DOI:10.1039/c7sc01905j
        ['[97*]-C1=NC=CC=C1-[99*]',  4.4, -1],  # 3-pyridylboronic acid DOI:10.1039/c7sc01905j
        ['O-C(=O)-C1=CC(-[97*])=C(-[99*])C=N1', 4.2, -1], # 5-boronopicolinic acid DOI:10.1039/c7sc01905j
        ['O-C(=O)-C1=CC=C(-[99*])C(-[97*])=N1', 4.2, -1], # 5-boronopicolinic acid DOI:10.1039/c7sc01905j
        ['C-C-C-N-C(=O)-C1=CC(-[97*])=C(-[99*])C=N1', 4.2,-1], # (6-propylcarbamoyl)pyridine-3-boronic acid DOI:10.1039/c7sc01905j
        ['C-C-C-N-C(=O)-C1=CC=C(-[99*])C(-[97*])=N1', 4.2,-1], # (6-propylcarbamoyl)pyridine-3-boronic acid DOI:10.1039/c7sc01905j
        ['O-C(=O)-C-[N+]1=CC=CC(-[99*])=C1-[97*]', 4.4,-1], # 3-borono-1-(carboxymethyl)pyridine DOI:10.1039/c7sc01905j
        ['O-C(=O)-C-[N+]1=CC=C(-[97*])C(-[99*])=C1', 4.4,-1], # 3-borono-1-(carboxymethyl)pyridine DOI:10.1039/c7sc01905j
        ['C-C-C(=O)-N-C1=CC=C(-[97*])C(-[99*])=C1', 8.3,-1], # 3-propionamidophenylboronic acid DOI:10.1039/c7sc01905j
        ['C-C-C(=O)-N-C1=CC=CC(-[99*])=C1-[97*]', 8.3,-1], # 3-propionamidophenylboronic acid DOI:10.1039/c7sc01905j
        ['CS(=O)(=O)C1=CC=C(-[99*])C=C1-[97*]', 7.1,-1], # 4-(methylsulfonyl)benzeneboronic acid DOI:10.1039/c7sc01905j pKa approx
        ['FC1=NC=CC(-[97*])=C1-[99*]', 6.3,-1]] # 2-Fluoro-3-pyridyl boronic acid DOI:10.1039/c7sc01905j 
        
        p = len(PBA_ortho)*[1/len(PBA_ortho)]
        index = np.random.choice(list(range(len(PBA_ortho))), p = p)
        PBA_mol, pKa, dummy = PBA_ortho[index]

        return Chem.MolFromSmiles(PBA_mol), str(pKa)

    if ph_subs == 'para':
       
        PBA_para=[['N-C1=CC(-[99*])=CC=C1-[97*]', 8.9,-1], # 3-Aminophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC=C(-[99*])C=C1', 8.8,-1], # Phenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC=C(-[99*])C=N1', 8.1,-1], # 3-Pyridinylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['C-C(=O)-C1=CC(-[99*])=CC=C1-[97*]', 8.0,-1], # 3-Acetophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC=C(-[99*])C=C1-C=O', 7.8,-1], # 3-Formylphenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[O-]-[N+](=O)-C1=CC(-[99*])=CC=C1-[97*]', 7.1,-1], # 3-Nitrophenylboronic acid 1 DOI:10.1016/j.tet.2004.08.051
        ['FC1=CC(-[99*])=C(F)C=C1-[97*]', 7.0,-1], # 2,5-Diflourophenylboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[O-]-[N+](=O)-C1=CC(-[99*])=C(F)C=C1-[97*]', 6.0,-1], # 2-Fluoro-5-nitrophenylboronic acid DOI:10.1016/j.tet.2004.08.051        
        ['[97*]-C-[N+]1=CC=C(-[99*])C=C1', 4.4,-1], # 4-Methylpyridineboronic acid DOI:10.1016/j.tet.2004.08.051
        ['[97*]-C1=CC=C(-[99*])C(-C=O)=C1', 7.31,-1], # 2-Formylphenilboronic acid DOI:10.3390/molecules25040799
        ['FC1=CC(-[99*])=CC=C1-[97*]', 7.5,-1], # 3-Fluorophenilboronic acid DOI:10.3390/molecules25040799
        ['FC(F)(F)C1=CC(-[99*])=CC=C1-[97*]', 7.85,-1], # 3-Metiltrifluorophenilboronic acid DOI:10.3390/molecules25040799
        ['[97*]-C1=CC=C(-[99*])C=C1-N-C(=O)-C=C', 8.2 ,-1], # 3-acrylamidophenylboronic acid (AAPBA) See the review DOI: 10.1039/C5CS00013K 
        ['[97*]-C1=CC=C(-[99*])C=C1-C(=O)-N-C-C-N-C(=O)-C=C',7.8,-1], # 4-(1,6-dioxo-2,5-diaza-7-oxamyl-)-phenylboronic acid (DDOPPA)
        #['[97*]-c1ccc2-[99*]-[#8]-[#6]-c2c1'. 7.2,0], # Benzoboroxole DOI: 10.1021/jo800788s Bonding with trans 4-6 diols prefered over trans 3-4 diol as in glucose.
        #['[#8]-[#6](=O)-c1cc2-[99*]-[#8]-[#6]-c2cc1-[97*]', 7.2,0], # Benzoboroxolebenzaldehide Binding to polimer throug carboxilic acid. Asummed to be similar to Benzoboroxole DOI: 10.1021/jo800788s
        ['FC1=NC(-[97*])=CC=C1-[99*]', 7.1,-1], # 2-fluoro-3-pyridylboronic acid or 2F-3-PyBA DOI: 10.1021/ol5036003
        ['[97*]-C1=NC=C(-[99*])C=N1', 6.2,-1], # pyrimidine-5-boronic acid DOI:10.1039/c7sc01905j
        ['[97*]-C1=CC=C(-[99*])C=N1', 4.4,-1],  # 3-pyridylboronic acid DOI:10.1039/c7sc01905j
        ['O-C(=O)-C-[N+]1=CC(-[99*])=CC=C1-[97*]', 4.4,-1], # 3-borono-1-(carboxymethyl)pyridine DOI:10.1039/c7sc01905j
        ['C-C-C(=O)-N-C1=CC(-[99*])=CC=C1-[97*]', 8.3, -1], # 3-propionamidophenylboronic acid DOI:10.1039/c7sc01905j
        ['FC1=NC(-[97*])=CC=C1-[99*]',6.3,-1]] # 2-Fluoro-3-pyridyl boronic acid DOI:10.1039/c7sc01905j

        p = len(PBA_para)*[1/len(PBA_para)]
        index = np.random.choice(list(range(len(PBA_para))), p = p)
        PBA_mol, pKa, dummy = PBA_para[index]

        return Chem.MolFromSmiles(PBA_mol), str(pKa)

if __name__ == "__main__":
    pass

