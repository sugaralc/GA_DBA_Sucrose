from sklearn.preprocessing import MinMaxScaler
import numpy as np
scaler = MinMaxScaler()
c1 = 0.5
c2 = 0.2
c3 = 0.15
c4 = 0.15

def score_function(energies,LogPs,pKas_max,sa_scores):
    data = []
    #for g,logp,pka,sa in zip(energies,LogPs,pKas_max,sa_scores):
        #data.append([g,logp,pka,sa])
        #data.append([g,pka,sa])
    for g,pka,sa in zip(energies,pKas_max,sa_scores):
        data.append([g,pka,sa])

    #Adding the extreme values to have the same scale in all the iterations
    #data.append([-30,-3,3,1])
    #data.append([1,6,10,10])
    data.append([-20,3,1])
    data.append([1,10,10])

    model = scaler.fit(data)
    scaled_data = model.transform(data)
    #print(scaled_data)
    scores = []
    for i in range(len(scaled_data)-2):
        G_scaled = scaled_data[i][0]
        #LogP_scaled = scaled_data[i][1]
        pKa_scaled = scaled_data[i][1]
        SA_scaled = scaled_data[i][2]
        LogP_scaled = 1.0 - LogP_gaussian_transformation(LogPs[i],-0.5,1.0)

        ind_score = (c1*G_scaled + c2*LogP_scaled + c3*pKa_scaled + c4*SA_scaled)
        scores.append(ind_score)
    
    return(scores)    

def LogP_gaussian_transformation(score: float, target: float, sigma: float) -> float:
    """Modifies a score to a fitness to a target using a gaussian distribution
    If the score matches the target value, the function evaluates to 1.
    The width of the distribution is controlled through sigma.
    :param score: the score to evaulate fitness for
    :param target: the target value to use for fitness evaluation
    :param sigma: the width of the distribution
    :return: the fitness evaluated as a gaussian
    """
    score = np.exp(-0.5 * np.power((score - target) / sigma, 2.0))
    return score
    