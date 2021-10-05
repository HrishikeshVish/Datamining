import pandas as pd
import copy
import numpy as np
import gc
pd.options.mode.chained_assignment = None


def discrete(no_bins):
    df = pd.read_csv('dating.csv')
    attributes = list(df.columns)
    attributes.remove('gender')
    attributes.remove('race')
    attributes.remove('race_o')
    attributes.remove('samerace')
    attributes.remove('field')
    attributes.remove('decision')
    continuous_vals  = attributes

    bins = {}
    for i in range(len(continuous_vals)):
        bins[continuous_vals[i]] = []
        if('age' in continuous_vals[i]):
            interval = (58-18)/no_bins
            switch_points = []
            start = 18
            while(start<=58):
                switch_points.append(start)
                start = round(start + interval,2)
            
            bins[continuous_vals[i]] = switch_points
        elif('pref_o' in continuous_vals[i] or '_important' in continuous_vals[i]):
            interval = (1)/no_bins
            switch_points = []
            start = 0
            while(start<=1):
                switch_points.append(start)
                start = round(start + interval,3)
            
            bins[continuous_vals[i]] = switch_points
        elif('correlate' in continuous_vals[i]):
            interval = (2)/no_bins
            switch_points = []
            start = -1
            while(start<=1):
                switch_points.append(start)
                start = round(start + interval,2)
            bins[continuous_vals[i]] = switch_points
        else:
            interval = (10)/no_bins
            switch_points = []
            start = 0
            while(start<=10):
                switch_points.append(start)
                start = round(start + interval,2)
            bins[continuous_vals[i]] = switch_points
        bins[continuous_vals[i]][0] -= 10
        bins[continuous_vals[i]][no_bins] **=2
    
    for i in continuous_vals:
        bin_range = bins[i]
        df[i] = pd.cut(df[i], bin_range, labels=range(no_bins))
    return df

def split(bins):
    
    df = discrete(bins)
    test_sample = df.sample(random_state=47, frac=0.2)
    indexes = list(test_sample.index)
    train_sample = df.drop(indexes)
    test_sample.reset_index(drop=True, inplace=True)
    train_sample.reset_index(drop=True, inplace=True)
    return test_sample, train_sample

non_cont = ['gender', 'race', 'race_o', 'samerace', 'field', 'decision']
def nbc(t_frac, no_bins):
    testing, training = split(no_bins)
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

no_bins = [2,5, 10, 50]
yet = [100, 200]
for i in [2,5,10,50,100,200]:
    print("Bin Size: %d"%(i))
    nbc(1, i)
    #split(i)

