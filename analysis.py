import pandas as pd
import numpy as np
from kmeans import kmeans
import copy
import json
from scipy.spatial.distance import cdist
dataset1 = pd.read_csv('digits-embedding.csv', header=None)
dataset1.columns = ['id', 'class', 'feature1', 'feature2']
dataset2 = dataset1[dataset1['class'].isin([2,4,6,7])]
dataset3 = dataset2[dataset2['class'].isin([6,7])]
print(dataset3)
results = {}
datacount = 0
for data in [dataset1, dataset2, dataset3]:
    datacount+=1
    temp = copy.copy(data)
    temp = temp.drop('id', axis=1)
    temp = temp.drop('class', axis=1)
    distances = cdist(data, data, metric='euclidean')
    results[datacount] = {}
    for K in [2,4,8,16,32]:
        wc_ssd, silCoef, nmi = kmeans(K, data, distances)
        results[datacount][K] = {}
        results[datacount][K]['wc'] = wc_ssd
        results[datacount][K]['sc'] = silCoef
        results[datacount][K]['nmi'] = nmi
with open('results.json', 'w') as fp:
    json.dump(results, fp)

        
