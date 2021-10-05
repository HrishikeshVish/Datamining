import pandas as pd
import numpy as np

non_cont = ['gender', 'race', 'race_o', 'samerace', 'field', 'decision']
def nbc(t_frac):
    df = pd.read_csv('dating-binned.csv')
    training = pd.read_csv('trainingSet.csv')
    train_mod = training.sample(random_state=47, frac = t_frac)
    testing = pd.read_csv('testSet.csv')

    train_mod.reset_index(drop=True, inplace=True)
    
    training = train_mod
    #testing = pd.concat([training, testing])
    #testing.reset_index(drop=True, inplace=True)
    positive_class = training.loc[training['decision'] == 1]
    negative_class = training.loc[training['decision'] == 0]

    prior = {}
    prior[1] = len(positive_class)/len(training)
    prior[0] = len(negative_class)/len(training)

    cond_prob_matrix_pos = {}
    cond_prob_matrix_neg = {}
    len_pos = len(positive_class)
    len_neg = len(negative_class)
    attribs = list(training.columns)
    laplace = {}
    for i in attribs:
        if(i == 'decision'):
            break
        attrib_values = list(set(list(training[i])))
        if i in non_cont:
            laplace[i] = len(attrib_values)
        else:
            laplace[i] = 5
        cond_prob_matrix_pos[i] = {}
        cond_prob_matrix_neg[i] = {}
        for j in attrib_values:
            cond_prob_matrix_pos[i][j] = (list(positive_class[i]).count(j)+1)/(laplace[i] + len_pos)
            cond_prob_matrix_neg[i][j] = (list(negative_class[i]).count(j)+1)/(laplace[i] + len_neg)

    count = 0
    for i in range(len(training)):
        init_prob_pos = prior[1]
        init_prob_neg = prior[0]
        for j in attribs:
            if(j == 'decision'):
                break
            init_prob_pos = init_prob_pos * cond_prob_matrix_pos[j][training[j][i]]
            init_prob_neg = init_prob_neg * cond_prob_matrix_neg[j][training[j][i]]
        frac_pos = init_prob_pos/(init_prob_pos + init_prob_neg)
        frac_neg = init_prob_neg/(init_prob_pos + init_prob_neg)
        if(frac_pos > frac_neg):
            decision = 1
        else:
            decision = 0
        
        if(decision == training['decision'][i]):
            count+=1
        
    print("Training Accuracy: %0.2f"%(count/len(training)))
    count = 0
    for i in range(len(testing)):
        init_prob_pos = prior[1]
        init_prob_neg = prior[0]
        for j in attribs:
            if(j == 'decision'):
                break
            try:
                init_prob_pos = init_prob_pos * cond_prob_matrix_pos[j][testing[j][i]]
            except:
                init_prob_pos = init_prob_pos * (1/laplace[j])
            try:
                init_prob_neg = init_prob_neg * cond_prob_matrix_neg[j][testing[j][i]]
            except:
                init_prob_neg = init_prob_neg * (1/laplace[j])
        frac_pos = init_prob_pos/(init_prob_pos + init_prob_neg)
        frac_neg = init_prob_neg/(init_prob_pos + init_prob_neg)
        if(frac_pos > frac_neg):
            decision = 1
        else:
            decision = 0
        if(decision == testing['decision'][i]):
            count+=1

    print("Testing Accuracy: %0.2f"%(count/len(testing)))
                
            
            
F = [0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1]        
for i in F:
    nbc(i)
