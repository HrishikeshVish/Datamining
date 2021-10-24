import numpy as np
import pandas as pd
from statistics import mean
import argparse
pd.options.mode.chained_assignment = None

df = pd.read_csv('dating-full.csv')
remove = list(range(6500, 6744))
df.drop(remove, inplace=True)
attributes = list(df.columns)
attributes = list(df.columns)
cleanAttribCount = 0
attribs_to_clean = ['race', 'race_o', 'field']
toLowerCount = 0
for attrib in attribs_to_clean:
    for i in range(len(df[attrib])):
        if(df[attrib][i].startswith("'")):
            df[attrib][i] = df[attrib][i].replace("'", '')
            cleanAttribCount +=1
        if(attrib == 'field'):
            if(any(ele.isupper() for ele in df[attrib][i])):
               toLowerCount +=1
               df[attrib][i] = df[attrib][i].lower()

gender = sorted(list(set(df['gender'])))
race = sorted(list(set(df['race'])))
race_o = sorted(list(set(df['race_o'])))
field = sorted(list(set(df['field'])))

gender = {item:gender.index(item) for item in gender}
race = {item:race.index(item) for item in race}
race_o = {item:race_o.index(item) for item in race_o}
field = {item:field.index(item) for item in field}

for i in range(len(df['gender'])):
    df['gender'][i] = gender[df['gender'][i]]
    df['race'][i] = race[df['race'][i]]
    df['race_o'][i] = race_o[df['race_o'][i]]
    df['field'][i] = field[df['field'][i]]

for index, row in df.iterrows():
    total1 = df['attractive_important'][index] + df['sincere_important'][index] + df['intelligence_important'][index] + df['funny_important'][index] + df['ambition_important'][index] + df['shared_interests_important'][index]
    total2 = df['pref_o_attractive'][index] + df['pref_o_sincere'][index] + df['pref_o_intelligence'][index] + df['pref_o_funny'][index] + df['pref_o_ambitious'][index] + df['pref_o_shared_interests'][index]
    
    df['attractive_important'][index] /=total1
    df['sincere_important'][index] /=total1
    df['intelligence_important'][index] /=total1
    df['funny_important'][index] /=total1
    df['ambition_important'][index] /=total1
    df['shared_interests_important'][index] /=total1

    df['pref_o_attractive'][index] /= total2
    df['pref_o_sincere'][index] /=total2
    df['pref_o_intelligence'][index] /=total2
    df['pref_o_funny'][index] /=total2
    df['pref_o_ambitious'][index] /=total2
    df['pref_o_shared_interests'][index] /=total2

attributes = list(df.columns)
attributes.remove('gender')
attributes.remove('race')
attributes.remove('race_o')
attributes.remove('samerace')
attributes.remove('field')
attributes.remove('decision')
continuous_vals  = attributes

def discrete(df, no_bins):

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
                start = round(start + interval,2)
            
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
        bin_count = []
        bin_count.append(list(df[i]).count(0))
        bin_count.append(list(df[i]).count(1))
        bin_count.append(list(df[i]).count(2))
        bin_count.append(list(df[i]).count(3))
        bin_count.append(list(df[i]).count(4))
        print(str(i)+':', bin_count)
    return df
df = discrete(df, 5)
test_sample = df.sample(random_state=25, frac=0.2)
indexes = list(test_sample.index)
train_sample = df.drop(indexes)

test_sample.to_csv('testSet_NBC.csv', index=False)
train_sample.to_csv('trainingSet_NBC.csv', index=False)


    
    
    
    
    

    
    
    
    
    




