import numpy as np
import pandas as pd
from statistics import mean
import argparse
import copy
pd.options.mode.chained_assignment = None

df = pd.read_csv('dating-full.csv')
remove = list(range(6500, 6744))
df.drop(remove, inplace=True)
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

gender_map = {}
race_map = {}
race_o_map = {}
field_map = {}

g_val = [0 for i in range(len(gender)-1)]
r_val = [0 for i in range(len(race)-1)]
ro_val = [0 for i in range(len(race_o)-1)]
f_val = [0 for i in range(len(field)-1)]

init_vals = [0 for i in range(6500)]

for i in range(len(gender)):
    gender_map[gender[i]] = copy.copy(g_val)
    if(i < len(gender) -1):
        gender_map[gender[i]][i] = 1
for i in range(len(race)):
    
    race_map[race[i]] = copy.copy(r_val)
    if(i <len(race)-1):
        df.insert(i+4, 'race'+str(i), init_vals, True)
        race_map[race[i]][i] = 1
for i in range(len(race_o)):
    
    race_o_map[race_o[i]] = copy.copy(ro_val)
    if(i<len(race_o)-1):
        df.insert(i+9, 'race_o'+str(i), init_vals, True)
        race_o_map[race_o[i]][i] = 1
for i in range(len(field)):
    
    field_map[field[i]] = copy.copy(f_val)
    if(i <len(field)-1):
        df.insert(i+17, 'field'+str(i), init_vals, True)
        field_map[field[i]][i] = 1

for i in range(len(df['gender'])):
    df['gender'][i] = gender_map[df['gender'][i]][0]
    raceVal = race_map[df['race'][i]]
    if(1 in raceVal):
        ind = raceVal.index(1)
        df['race'+str(ind)][i] = 1
    raceoVal = race_o_map[df['race_o'][i]]
    if(1 in raceoVal):
        ind = raceoVal.index(1)
        df['race_o'+str(ind)][i] = 1
    fieldVal = field_map[df['field'][i]]
    if(1 in fieldVal):
        ind = fieldVal.index(1)
        df['field'+str(ind)][i] = 1
    
    #df['race'][i] = race_map[df['race'][i]]
    #df['race_o'][i] = race_o_map[df['race_o'][i]]
    #df['field'][i] = field_map[df['field'][i]]
df.drop(columns=['race','race_o', 'field'], inplace=True)
print("Mapped vector for female in column gender:",str(gender_map['female'])+'.')
print("Mapped vector for Black/African American in column race: :", str(race_map['Black/African American'])+'.')
print("Mapped vector for Other in column race_o::", str(race_o_map['Other'])+'.')
print("Mapped vector for economics in column field:", str(field_map['economics'])+'.')


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

test_sample = df.sample(random_state=45, frac=0.2)
indexes = list(test_sample.index)
train_sample = df.drop(indexes)
test_sample.to_csv('testSet.csv', index=False)
train_sample.to_csv('trainingSet.csv', index=False)


    
    
    
    
    

    
    
    
    
    




