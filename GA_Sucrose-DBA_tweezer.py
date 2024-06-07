import heapq
import random
import shutil
from pathlib import Path
from typing import List

import numpy as np
import submitit
from rdkit import Chem
from scipy.stats import rankdata
from tabulate import tabulate

import crossover4DBA as co
import filters
#import mutate as mu
#from catalyst import ts_scoring
from Gibbs_Reaction_Free_Energy import Free_energy_scoring
from Gibbs_Reaction_Free_Energy.utils import Individual
from sa import neutralize_molecules, sa_target_score_clipped, calculateScore
import os
from DBA_GA_utils import core_linker_frag, dummy2BO2, tweezer_classification
from Score_function import score_function
from dba_mututation import DBA_mutation

SLURM_SETUP = {
    "slurm_partition": "FULL",
    "timeout_min": 500,
    "slurm_array_parallelism": 16,
}


def slurm_scoring(sc_function, population, ids, cpus_per_task=4, cleanup=False):
    """Evaluates a scoring function for population on SLURM cluster

    Args:
        sc_function (function): Scoring function which takes molecules and id (int,int) as input
        population (List): List of rdkit Molecules
        ids (List of Tuples of Int): Index of each molecule (Generation, Individual)

    Returns:
        List: List of results from scoring function
    """
    executor = submitit.AutoExecutor(
        folder="scoring_tmp",
        slurm_max_num_timeout=0,
        #cluster="debug"
    )
    executor.update_parameters(
        name=f"sc_g{ids[0][0]}",
        cpus_per_task=cpus_per_task,
        tasks_per_node=1,
        #slurm_mem_per_cpu="6GB",
        slurm_mem="300GB",
        timeout_min=SLURM_SETUP["timeout_min"],
        slurm_partition=SLURM_SETUP["slurm_partition"],
        slurm_array_parallelism=SLURM_SETUP["slurm_array_parallelism"],
    )
    #job_env = submitit.JobEnviroment()
    #ntasks = job_env.num_tasks
    #os.environ[]
    args = [cpus_per_task for p in population]
    jobs = executor.map_array(sc_function, population, ids, args)

    results = [
        catch(job.results, handle=lambda e: (np.nan, np.nan)) for job in jobs
    ]  # catch submitit exceptions and return same output as scoring function (np.nan, None) for (energy, geometry)
    if cleanup:
        shutil.rmtree("scoring_tmp")
    return results


def catch(func, *args, handle=lambda e: e, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(e)
        return handle(e)


def make_initial_population(population_size, directory, rand=True): #Here I should use my implementation with xyz files to Chem.Mol
    if rand:
        sample = heapq.nlargest(population_size, os.listdir(directory), key=lambda L: random.random())
    else:
        sample = os.listdir(directory)
    population = [core_linker_frag(directory,xyzfile) for xyzfile in sample]
    
    return population


def calculate_normalized_fitness(scores):
    sum_scores = sum(scores)
    normalized_fitness = [score / sum_scores for score in scores]

    return normalized_fitness


def calculate_fitness(scores, minimization=False, selection="roulette", selection_pressure=None):
    if minimization:
        scores = [-s for s in scores]
    if selection == "roulette":
        fitness = scores
    elif selection == "rank":
        scores = [float("-inf") if np.isnan(x) else x for x in scores]  # works for minimization
        ranks = rankdata(scores, method="ordinal")
        n = len(ranks)
        if selection_pressure:
            fitness = [
                2 - selection_pressure + (2 * (selection_pressure - 1) * (rank - 1) / (n - 1))
                for rank in ranks
            ]
        else:
            fitness = [r / n for r in ranks]
    else:
        raise ValueError(
            f"Rank-based ('rank') or roulette ('roulette') selection are available, you chose {selection}."
        )

    return fitness


def make_mating_pool(population, fitness, mating_pool_size):
    mating_pool = []
    for i in range(mating_pool_size):
        mating_pool.append(random.choices(population, weights=fitness, k=1)[0])
    return mating_pool


def reproduce(mating_pool, population_size, mutation_rate, molecule_filter, generation):
    new_population = []
    counter = 0
    while len(new_population) < population_size:
        if random.random() > mutation_rate:
            parent_A = random.choice(mating_pool)
            parent_B = random.choice(mating_pool)
            tempnew_child = co.crossover(parent_A.rdkit_mol, parent_B.rdkit_mol)
            if co.mol_OK(tempnew_child) and co.DBA_crossover_OK(tempnew_child):
                idx_glcyl, glcylpatt = tweezer_classification(tempnew_child,'Glc')
                idx_frcyl, frcylpatt = tweezer_classification(tempnew_child,'Frc')
                new_child = DBA_mutation(tempnew_child,
                                         idx_glcyl,glcylpatt,
                                         idx_frcyl,frcylpatt
                                         )
                if new_child != None:
                    idx = (generation, counter)
                    counter += 1
                    new_child = Individual(rdkit_mol=new_child, idx=idx)
                    new_population.append(new_child)
        else:
            parent = random.choice(mating_pool)
            try:
                idx_glcyl, glcylpatt = tweezer_classification(parent.rdkit_mol,'Glc')
                idx_frcyl, frcylpatt = tweezer_classification(parent.rdkit_mol,'Frc')
            except:
                print(f'WARNING: The molecule {parent.smiles} could not be classified for mutation operation. mutated_child = None.')
                mutated_child = None
            else:
                mutated_child = DBA_mutation(parent.rdkit_mol,
                                         idx_glcyl, glcylpatt,
                                         idx_frcyl, frcylpatt,
                                         )
            if mutated_child != None:
                idx = (generation, counter) #Here idx is a string and is printed like (0, 1)
                counter += 1                #for the second individual (1) in the firs generation (0)
                mutated_child = Individual( #GALC change 20/02/24
                    rdkit_mol=mutated_child,
                    idx=idx,
                )
                new_population.append(mutated_child)
    return new_population


def sanitize(population, population_size, prune_population):
    if prune_population:
        sanitized_population = []
        for ind in population:
            if ind.smiles not in [si.smiles for si in sanitized_population]:
                sanitized_population.append(ind)
    else:
        sanitized_population = population

    sanitized_population.sort(
        key=lambda x: float("inf") if np.isnan(x.score) else x.score
    )  # np.nan is highest value, works for minimization of score

    new_population = sanitized_population[:population_size]
    return new_population  # selects individuals with lowest values


def reweigh_scores_by_sa(population: List[Chem.Mol], scores: List[float]) -> List[float]:
    """Reweighs scores with synthetic accessibility score
    :param population: list of RDKit molecules to be re-weighted
    :param scores: list of docking scores
    :return: list of re-weighted docking scores
    """
    sa_scores = [sa_target_score_clipped(p) for p in population]
    return sa_scores, [
        ns * sa for ns, sa in zip(scores, sa_scores)
    ]  # rescale scores and force list type
    #I should understand and decide if the scores should be already weigthed at this point, or 
    #or make the weigthing at this point. GALC 20/02/24.

def print_results(population, fitness, generation):
    print(f"\nGeneration {generation+1}", flush=True)
    print(
        tabulate(
            [
                [ind.idx, fit, ind.score, ind.energy, ind.LogP, ind.pKa_max, ind.sa_score, ind.smiles]
                for ind, fit in zip(population, fitness)
            ],
            headers=[  #Modify here to include the new terms in the fitness function. GALC change 20/02/24
                "idx",
                "normalized fitness",
                "score",
                "Free Energy",
                "LogP",
                "pKa max",
                "sa score",
                "smiles",
            ],
        ),
        flush=True,
    )


def GA(args):
    (
        population_size,
        #file_name,
        molecules_directory,
        scoring_function,
        generations,
        mating_pool_size,
        mutation_rate,
        scoring_args,
        prune_population,
        seed,
        minimization,
        selection_method,
        selection_pressure,
        molecule_filters,
        path,
    ) = args

    np.random.seed(seed)
    random.seed(seed)

    generations_file = Path(path) / "generations.gen"
    generations_list = []

    molecules = make_initial_population(population_size, molecules_directory, rand=True)
    
    #print(len(smiles))
    #molecules = [Chem.MolFromSmiles(mol) for mol in smiles] #Here write the function of ligands assigning the values of pKa
    #for mol in molecules:
    #    mol.SetProp('pKaglcyl',str(8.8))
    #    mol.SetProp('pKafrcyl',str(8.8))
    

    # write starting popultaion
    pop_file = Path(path) / "starting_pop.smi"
    with open(str(generations_file.resolve()), "w+") as f:
        f.writelines([str(Chem.MolToSmiles(m) for m in molecules)])

    #for molecule in molecules:
    #    print(Chem.MolToSmiles(molecule,kekuleSmiles=True))

    ids = [(0, i) for i in range(len(molecules))]
    results = slurm_scoring(scoring_function, molecules, ids, cpus_per_task=38)
    #results = []
    #for i in range(population_size):
    #    results.append([np.random.uniform(low=-15, high=3),
    #                    np.random.uniform(low=1, high=5)
    #                    ])
    
    energies = []
    LogPs = []
    for res in results:
        energies.append(res[0][0])
        LogPs.append(res[0][1])

    #energies = [res[0] for res in results]
    #LogPs = [res[1] for res in results]

    pKas_max = []
    for mol in molecules:
        pKa1 = float(mol.GetProp('pKaglcyl'))
        pKa2 = float(mol.GetProp('pKafrcyl'))
        if pKa1 < pKa2:
            pKas_max.append(pKa2)
        else:
            pKas_max.append(pKa1)

    mols4SA = [dummy2BO2(mol) for mol in molecules]
    mols_neutral = neutralize_molecules(mols4SA)
    sa_scores = []
    for mol in mols_neutral:
        temp_sascore = calculateScore(mol)
        sa_scores.append(temp_sascore)

    scores = score_function(energies,LogPs,pKas_max,sa_scores)
    #prescores = [energy - 100 for energy in energies]
    #sa_scores, scores = reweigh_scores_by_sa(neutralize_molecules(molecules), prescores)

    population = [    
        Individual(   
            idx=idx,  
            rdkit_mol=mol, 
            score=score,   
            energy=energy,
            LogP=logp,
            pKa_max=pka_max,
            sa_score=sa_score,
        )
        for idx, mol, score, energy, logp, pka_max ,sa_score in zip(
            ids, molecules, scores, energies, LogPs, pKas_max, sa_scores
        )
    ]
    population = sanitize(population, population_size, False)

    fitness = calculate_fitness(
        [ind.score for ind in population], 
        minimization,                      
        selection_method,                  
        selection_pressure,                
    )
    fitness = calculate_normalized_fitness(fitness)

    score_best = 1000
    count_score_best= 0
    for ind in population:
        if ind.score < score_best:
            score_best = ind.score

    print_results(population, fitness, -1)
    print("Generation, score_best, count_score_best")
    print(0, score_best, count_score_best)

    for generation in range(generations):
        if count_score_best > 10:
            print("GA converged. count_best_score =",count_score_best)
            break

        mating_pool = make_mating_pool(population, fitness, mating_pool_size)

        new_population = reproduce(
            mating_pool,
            population_size-mating_pool_size,
            mutation_rate,
            molecule_filters,
            generation + 1,
        )

        #for mol in new_population:
        #    print(mol.smiles)

        new_results = slurm_scoring(
            scoring_function,
            [ind.rdkit_mol for ind in new_population],
            [ind.idx for ind in new_population],
            cpus_per_task=38
        )
        #new_results = []
        #for i in range(len(new_population)):
        #    new_results.append([np.random.uniform(low=-15, high=3),
        #                    np.random.uniform(low=1, high=5)
        #                    ])

        new_energies = []
        new_LogPs = []
        for res in new_results:
            new_energies.append(res[0][0])
            new_LogPs.append(res[0][1])

        #new_energies = [res[0] for res in new_results]
        #new_LogPs = [res[1] for res in new_results]
        

        new_pKas_max = []
        for new_mol in new_population:
            new_mol_rdkit = new_mol.rdkit_mol
            pKa1 = float(new_mol_rdkit.GetProp('pKaglcyl'))
            pKa2 = float(new_mol_rdkit.GetProp('pKafrcyl'))
            if pKa1 < pKa2:
                new_pKas_max.append(pKa2)
            else:
                new_pKas_max.append(pKa1)

        mols4SA = [dummy2BO2(new_mol.rdkit_mol) for new_mol in new_population]
        new_mols_neutral = neutralize_molecules(mols4SA)
        new_sa_scores = [calculateScore(new_mol_c0) for new_mol_c0 in new_mols_neutral]
        new_scores = score_function(new_energies,new_LogPs,new_pKas_max,new_sa_scores)

        #new_prescores = [energy - 100 for energy in new_energies]
        #new_sa_scores, new_scores = reweigh_scores_by_sa(
        #    neutralize_molecules([ind.rdkit_mol for ind in new_population]),
        #    new_prescores,
        #)

        for ind, score, energy, logp, pka_max, sa_score in zip(
            new_population,
            new_scores,
            new_energies,
            new_LogPs,
            new_pKas_max,
            new_sa_scores,
        ):
            ind.score = score
            ind.energy = energy
            ind.LogP = logp
            ind.pKa_max = pka_max
            ind.sa_score = sa_score

        population = sanitize(population + new_population, population_size, prune_population)

        fitness = calculate_fitness(
            [ind.score for ind in population],
            minimization,
            selection_method,
            selection_pressure,
        )
        fitness = calculate_normalized_fitness(fitness)

        generations_list.append([ind.idx for ind in population])
        print_results(population, fitness, generation)

        score_best_new = 1000

        for ind in population:
            if ind.score < score_best_new:
                score_best_new = ind.score
       
        if score_best_new < score_best:
            score_best = score_best_new
            count_score_best = 0
        else:
            count_score_best += 1
            
        print("Generation, score_best, count_score_best")
        print(generation+1, score_best, count_score_best)


    with open(str(generations_file.resolve()), "w+") as f:
        f.writelines(str(generations_list))


if __name__ == "__main__":

    package_directory = Path(__file__).parent.resolve()
    print(package_directory)

    co.average_size = 40.022840038202613
    co.size_stdev = 4.230907997270275
    population_size = 32
    molecules_directory = package_directory / "linkers32_4tunning"
    #file_name = package_directory / "ZINC_amines.smi"
    scoring_function = Free_energy_scoring 
    generations = 50
    mating_pool_size = round(population_size*0.5)
    mutation_rate = 0.8
    scoring_args = None
    prune_population = True

    from datetime import datetime
    t = datetime.now().time()
    seconds = int((t.hour * 60 + t.minute) * 60 + t.second)
    seed = seconds
    print('seed=',seed)

    minimization = True
    selection_method = "rank"
    selection_pressure = 1.8
    molecule_filters = filters.get_molecule_filters(
        ["MBH"], package_directory / "filters/alert_collection.csv"
    ) 
    # file_name = argv[-1]

    path = "."

    args = [
        population_size,
        #file_name,
        molecules_directory,
        scoring_function,
        generations,
        mating_pool_size,
        mutation_rate,
        scoring_args,
        prune_population,
        seed,
        minimization,
        selection_method,
        selection_pressure,
        molecule_filters,
        path,
    ]

    # Run GA
    GA(args)

#smiles = ['[98*]C1=CC=CC2=C1CC1=CC3=C(C=C1C2)CCC=C3C1=CC([99*])=CC=C1',
#              '[98*]C1=C([C@H]2CC=CC3=C2CC2=C3CC3=C([99*])C=CC=C3C2)C=CN=C1F',
#              '[98*]C1=CC=CC([C@@H]2CC[C@]3(CC[C@H](C4=CC([99*])=CC=C4)C(=C)C3)C2)=C1',
#              '[98*]C1=CC=CC=C1[C@@H]1CC[C@@]2(C=C(C3=CC([99*])=CC=C3)CC2)C1',
#              '[98*]c1ccc([C@@H]2C=CC[C@]3(CC[C@@H](c4cccc([99*])c4)C(=C)C3)C2)cc1',
#              '[98*]C1=CC=CC([C@@H]2CC[C@]3(CC[C@H](C4=CC([99*])=CC=C4)C(=C)C3)C2)=C1',
#              '[98*]C1=CC=CC([C@H]2CCC3=C(CC4=CC5=C(C=C34)C([99*])=CC=C5)C2)=C1',
#              '[98*]C1=CC=CC(/C=C2/CC[C@]3(CC[C@@H](C4=CC([99*])=CC=C4)CC3)C2)=C1',
#              '[98*]C1=CC=CC(/C=C2/CCC[C@]23CC[C@@H](C2=CC([99*])=CC=C2)CC3)=C1',
 #             '[98*]C1=CC=CC=C1C1=CC2=C(C(C3=CC([99*])=CC=C3)=CC=C2)C2=C1C=CC1=C2C=CC=C1',
 #             '[98*]C1=CC=CC([C@@H]2CC[C@]3(CC[C@H](C4=CC([99*])=CC=C4)CC3)C2)=C1',
 #             '[98*]C1=CC=CC2=C1C1=CC3=C(C=CC(C4=CC([99*])=CC=C4)=C3)C=C1CC2',
#              '[98*]C1=CC=CC([C@@H]2CC3=C(C=C4CC5=C(C=CC=C5[99*])CC4=C3)C2)=C1',
#              '[98*]C1=CC=CC=C1C1=CC2=C(C=C1)C1=C(C=CC=C1C1=CC([99*])=CC=C1)C2',
#              '[98*]C1=CC=CC2=C1CC1=C3C[C@H](C4=CC([99*])=CC=C4)CC3=CC=C1C2',
#              '[98*]C1=CC=CC([C@H]2C3=C(C=CC=C3)C3=C2C=CC(C2=CC([99*])=CC=C2)=C3)=C1',
 #             '[98*]C1=CC=CC([C@@H]2C=CC3=C(CC4=C3CC3=C4C([99*])=CC=C3)C2)=C1',
 #             '[98*]C1=CC=CC([C@H]2CC3=C(C=C4C[C@H](C5=CC([99*])=CC=C5)CC4=C3)C2)=C1',
 #             '[98*]C1=CC=CC2=C1C1=C(C=CC3=C1C=CC(C1=CC([99*])=CC=C1)=C3)CC2',
 #             '[98*]C1=CC=CC([C@@H]2CCC[C@H]([C@@H]3CCC[C@H](C4=CC([99*])=CC=C4)C3)C2)=C1',
 #             '[98*]C1=CC=CC([C@H]2C=CC3=CC(C4=CC([99*])=CC=C4)=CC=C32)=C1',
 #             '[98*]C1=CC=CC([C@H]2CCC3=C2C=CC2=CC4=C(C=C32)C([99*])=CC=C4)=C1',
 #             '[98*]C1=CC=CC([C@H]2CC[C@]3(C=CC[C@H](C4=CC([99*])=CC=C4)C3)CC2)=C1',
 #             '[98*]C1=CC=CC=C1C1=CCC2=C(C3=CC([99*])=CC=C3)C=CC3=CC=CC1=C23',
 #             '[98*]C1=CC=CC=C1C1=C2CC=CC3=CC=C(C4=CC([99*])=CC=C4)C(=C32)C=C1',
 #             '[98*]C1=CC=CC([C@@H]2C=CC3=C4CC5=C(C([99*])=CC=C5)C4=CC=C3C2)=C1',
 #             '[98*]C1=CC=CC(C2=CC=CC3=C2C2=C(C=C(C4=CC([99*])=CC=C4)C=C2)C3)=C1',
 #             '[98*]C1=CC=CC(C2=CC3=C(C=C2)CC2=C3C=C(C3=CC([99*])=CC=C3)C=C2)=C1',
 #             '[98*]C1=CC=CC2=C1CC1=CC3=C(C=C1C2)CCC=C3C1=CC([99*])=CC=C1',
 #             '[98*]C1=CC=CC([C@@H]2C[C@@H]3CC[C@@H](C4=CC([99*])=CC=C4)C[C@@H]3C2)=C1',
 #             '[98*]C1=CC=CC=C1C1=CC2=C(C=C1)C(C1=CC([99*])=CC=C1)=C1C=CC=CC1=C2',
 #             '[98*]C1=CC=CC([C@@H]2C=CC3=C(C=C4C=CC5=C(C=CC=C5[99*])C4=C3)C2)=C1',
 #             '[98*]C1=CC=CC([C@H]2CC[C@@H](CC3=CC([99*])=CC=C3)C[C@H]2C)=C1',
 #             '[98*]C1=CC=CC(C2=CC3=C(C=C2)C=CC2=C3C3=C(C=C2)C(C2=CC([99*])=CC=C2)=CC=C3)=C1',
 #             '[98*]C1=CC=CC(C2=CC=CC3=C2C2=C(C=C3)C(C3=CC([99*])=CC=C3)=CC=C2)=C1',
 #             '[98*]C1=CC=CC([C@@H]2CC3=C(C=C4CC5=C(C=CC=C5[99*])CC4=C3)C2)=C1',
 #             '[98*]C1=CC=CC([C@@H]2C=CC3=C(C2)C2=C(C=C3)C3=C(C=C2)C([99*])=CC=C3)=C1',
 #             '[98*]C1=CC=CC2=C1C=C1C3=C(CC[C@@H](C4=CC([99*])=CC=C4)C3)CC1=C2',
 #             '[98*]C1=CC=CC=C1C1=CC2=C(CC1)CC1=C2CC2=C1C=CC=C2[99*]',
 #             '[98*]C1=CC=CC=C1C1=CC=CC2=C1CC1=C2C(C2=CC([99*])=CC=C2)=CC=C1',
 #             '[98*]C1=CC=C(CCC2=CCC3=C(C[N+]4=CC=C([99*])C=C4)C=CC4=CC=CC2=C43)C=C1C=O',
 #             '[98*]c1cnccc1[C@@H]1Cc2cc3c(cc2C1=O)Cc1c([99*])cccc1C3',
 #             '[98*]c1cccc([S@@H]2C=Cc3c(ccc4c3Cc3cccc([99*])c3-4)C2)c1',
 #             '[98*]C1=CC=CC([C@@H]2CCC[C@H]([C@@H]3CCC(=O)[C@@H](C4=CC([99*])=CC=C4)C3)C2)=C1',
 #             '[98*]C1=CC=CC([SH]2CC[C@]23CC[C@H](C2=CC([99*])=CC=C2)C(=C)C3)=C1',
 #             '[98*]C1=CC=CC([C@@H](C)CC[C@]2(C)C=CC[C@H](C[N+]3=CC=CC([99*])=C3F)C2)=C1',
 #             '[98*]C1=CC=[N+](C[C@@H]2C=CC3=C(CC4=C3CC3=CC=CC([99*])=C34)C2)C=C1',
 #             '[98*]C1=CC=C(C(=O)NCCC)[N+](C[C@H]2CC=CC3=C2CC2=C3CC3=C([99*])C=CC=C3C2)=C1',
 #             '[98*]C1=CC=C[N+](CCC2=CC=CC3=CC(C[N+]4=CN=CC([99*])=C4)C(C(O)C#N)=C32)=C1',
 #             '[98*]C1=CC=C(C(=O)NCCC)[N+](CC2=CC3=C(CC2)CC2=C3CC3=C([99*])C=CC=C32)=C1',
 #             '[98*]C1=CC=C(C(=O)NCCC)[N+](C[C@@H]2CC[C@]3(CC[C@@H](C4=NC=C([99*])C=N4)CC3)C2)=C1',
 #             '[98*]C1=CC=C(Br)C=C1[C@@H]1CC[C@@]2(C=C(C3=CC([99*])=CC=C3)C(=O)C2)C1',
 #             '[98*]C1=CC=CC([C@]2(F)CC3=CC4=C(C=C3C2)C[C@H](C[N+]2=CC([99*])=CC=C2C(=O)O)C4)=C1',
 #             '[98*]C1=CC=CC([C@@H]2CC3=CC4=C(C=C3CN2)CC2=C([99*])C=CC=C2C4)=C1',
 #             '[98*]C1=CC=C(C2=CC(Cl)=C3CC4=C(C3=C2)C(Br)=C(C[N+]2=CC([99*])=CC=C2C(=O)NCCC)C=C4)C(C(F)(F)F)=C1',
 #             '[98*]C1=C([C@@H]2C=CC3=CC=C4C5=CC=CC([99*])=C5C=CC4=C3C2=O)C=CN=C1F',
 #             '[98*]C1=C(C=O)C=CC2=C1CC1=C3C[C@H](C[N+]4=CC([99*])=CC=C4C(=O)O)CC3=CC=C1C2',
 #             '[98*]C1=CC=CC([C@@H]2C=CC3=CC=C4C5=CC=CC([99*])=C5C=CC4=C3C2=O)=C1',
 #             '[98*]C1=C(C(F)(F)F)C=CC2=C1C1=CC3=CC(C[N+]4=CC([99*])=CC=C4C(=O)O)=CC=C3C=C1CC2',
 #             '[98*]C1=CC=C(C(=O)NCCC)[N+](C[C@H]2CC[C@@H](CC3=NC(C(=O)O)=CC=C3[99*])C[C@H]2C)=C1',
 #             'C[NH+](C)Cc1ccc(cc1[99*])-c1ccc2cccc3C(=CCc1c23)c1ccccc1[98*]'
 #             ]
