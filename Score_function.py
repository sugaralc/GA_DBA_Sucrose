from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
c1 = 0.5
c2 = 0.2
c3 = 0.15
c4 = 0.15

def score_function(energies,LogPs,pKas_max,sa_scores):
    data = []
    for g,logp,pka,sa in zip(energies,LogPs,pKas_max,sa_scores):
        data.append([g,logp,pka,sa])

    model=scaler.fit(data)
    scaled_data=model.transform(data)
    scores = []
    for ind_scaled in scaled_data:
        G_scaled = ind_scaled[0]
        LogP_scaled = ind_scaled[1]
        pKa_scaled = ind_scaled[2]
        SA_scaled = ind_scaled[3]

        ind_score = (c1*G_scaled + c2*LogP_scaled + c3*pKa_scaled + c4*SA_scaled)
        scores.append(ind_score)
    
    return(scores)    