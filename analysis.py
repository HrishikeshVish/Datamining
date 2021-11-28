import pandas as pd
import numpy as np
from kmeans import kmeans
import copy
import json
from scipy.spatial.distance import cdist
dataset1 = pd.read_csv('digits-embedding.csv', header=None)
dataset1.columns = ['id', 'class', 'feature1', 'feature2']
dataset2 = dataset1[dataset1['class'].isin([2,4,6,7])]
dataset2.reset_index(inplace=True, drop=True)
for i in range(len(dataset2['id'])):
    dataset2['id'][i] = i
dataset3 = dataset2[dataset2['class'].isin([6,7])]
dataset3.reset_index(inplace=True, drop=True)
for i in range(len(dataset3['id'])):

    dataset3['id'][i] = i

results = {}
datacount = 0
for data in [dataset1, dataset2, dataset3]:
    datacount+=1
    temp = copy.copy(data)
    temp = temp.drop('id', axis=1)
    temp = temp.drop('class', axis=1)
    distances = cdist(temp, temp, metric='euclidean')
    results[datacount] = {}
    for K in [2, 4, 8, 16,32]:
        wc_ssd, silCoef, nmi = kmeans(K, data, distances_valid=distances, print_epoch=True)
        results[datacount][K] = {}
        results[datacount][K]['wc'] = wc_ssd
        results[datacount][K]['sc'] = silCoef
        results[datacount][K]['nmi'] = nmi

with open('results.json', 'w') as fp:
    json.dump(results, fp)


