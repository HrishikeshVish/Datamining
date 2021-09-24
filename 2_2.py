import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dating.csv")

rating_attribs = ['attractive_partner', 'sincere_partner', 'intelligence_parter',
'funny_partner', 'ambition_partner', 'shared_interests_partner']
decisions = list(df['decision'])
mainMapping = {}
for i in rating_attribs:
    col = list(df[i])
    #print(col)
    zipped = list(zip(col, decisions))
    #print(zipped)
    mainMapping[i] = {}
    distinct = list(set(col))
    #print(distinct)
    mapping = {}
    for j in distinct:
        total = col.count(j)
        filter1 = [t for t in zipped if t[0] == j]
        filter2 = [t for t in filter1 if t[1] == 1]
        frac = len(filter2)
        mapping[j] = frac/total
    mainMapping[i] = mapping
for i in mainMapping.keys():
    x = np.array(list(mainMapping[i].keys()))
    y = np.array(list(mainMapping[i].values()))
    plt.scatter(x,y)
    plt.xlabel(i)
    plt.ylabel("Success Rate")
    plt.savefig(i+'.png')
    plt.show()
    
    
