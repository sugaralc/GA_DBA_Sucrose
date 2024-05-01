from sklearn.preprocessing import MinMaxScaler
import numpy as np
scaler = MinMaxScaler()
c1 = 0.5
c2 = 0.2
c3 = 0.15
c4 = 0.15

def score_function(energies,LogPs,pKas_max,sa_scores):
    data = []
    for g,logp,pka,sa in zip(energies,LogPs,pKas_max,sa_scores):
        data.append([g,logp,pka,sa])

    #Adding the extreme values to have the same scale in all the iterations
    data.append([-15,-3,3,1])
    data.append([3,6,10,10])

    model = scaler.fit(data)
    scaled_data = model.transform(data)
    #print(scaled_data)
    scores = []
    for i in range(len(scaled_data)-2):
        G_scaled = scaled_data[i][0]
        LogP_scaled = scaled_data[i][1]
        pKa_scaled = scaled_data[i][2]
        SA_scaled = scaled_data[i][3]

        ind_score = (c1*G_scaled + c2*LogP_scaled + c3*pKa_scaled + c4*SA_scaled)
        scores.append(ind_score)
    
    return(scores)    