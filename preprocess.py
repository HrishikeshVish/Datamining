import numpy as np
import pandas as pd
from statistics import mean
pd.options.mode.chained_assignment = None

df = pd.read_csv('dating-full.csv')

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
print("Quotes removed from", cleanAttribCount, "cells.")
print("Standardized", toLowerCount, "cells to lower case.")

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
print("Value assigned for male in column gender:",str(gender['male'])+'.')
print("Value assigned for European/Caucasian-American in column race:", str(race['European/Caucasian-American'])+'.')
print("Value assigned for Latino/Hispanic American in column race o:", str(race_o['Latino/Hispanic American'])+'.')
print("Value assigned for law in column field:", str(field['law'])+'.')

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

print("Mean of attractive_important: %.2f."%mean(df['attractive_important']))
print("Mean of sincere_important: %.2f."%mean(df['sincere_important']))
print("Mean of intelligence_important: %.2f."%mean(df['intelligence_important']))
print("Mean of funny_important: %.2f."%mean(df['funny_important']))
print("Mean of ambition_important: %.2f."%mean(df['ambition_important']))
print("Mean of shared_interests_important: %.2f."%mean(df['shared_interests_important']))

print("Mean of pref_o_attractive: %.2f."%mean(df['pref_o_attractive']))
print("Mean of pref_o_sincere: %.2f."%mean(df['pref_o_sincere']))
print("Mean of pref_o_intelligence: %.2f."%mean(df['pref_o_intelligence']))
print("Mean of pref_o_funny: %.2f."%mean(df['pref_o_funny']))
print("Mean of pref_o_ambitious: %.2f."%mean(df['pref_o_ambitious']))
print("Mean of pref_o_shared_interests: %.2f."%mean(df['pref_o_shared_interests']))

df.to_csv('dating.csv', index=False)

    
    
    
    
    

    
    
    
    
    




