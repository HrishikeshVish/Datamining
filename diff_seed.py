import pandas as pd
import numpy as np
from kmeans import kmeans
import copy
import json
from scipy.spatial.distance import cdist
from statistics import mean, stdev
import math
import random
import matplotlib.pyplot as plt
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
K_vals = [4,8,16,32]
for data in [dataset1, dataset2, dataset3]:
    datacount+=1
    temp = copy.copy(data)
    temp = temp.drop('id', axis=1)
    temp = temp.drop('class', axis=1)
    distances = cdist(temp, temp, metric='euclidean')
    results[datacount] = {}
    for K in K_vals:
        results[datacount][K] = {}
        results[datacount][K]['wc'] = []
        results[datacount][K]['sc'] = []
        results[datacount][K]['nmi'] = []
        for i in range(10):
            seed_val = random.randint(0,100)
            wc_ssd, silCoef, nmi = kmeans(K, data, distances, seed_value=seed_val)
            results[datacount][K]['wc'].append(wc_ssd)
            results[datacount][K]['sc'].append(silCoef)
            results[datacount][K]['nmi'].append(nmi)
        results[datacount][K]['wc'] = [mean(results[datacount][K]['wc']), stdev(results[datacount][K]['wc'])/math.sqrt(10)]
        results[datacount][K]['sc'] = [mean(results[datacount][K]['sc']), stdev(results[datacount][K]['sc'])/math.sqrt(10)]
with open('results_10.json', 'w') as fp:
    json.dump(results, fp)
with open('results_10.json') as fp:
    results = json.load(fp)
"""
wc_mean_vals = []
sc_mean_vals = []
wc_std_vals = []
sc_std_vals = []
for i in K_vals:
    wc_mean_vals.append(results[1][i]['wc'][0])
    wc_std_vals.append(results[1][i]['wc'][1])
    sc_mean_vals.append(results[1][i]['sc'][0])
    sc_std_vals.append(results[1][i]['sc'][1])

plt.errorbar(K_vals, wc_mean_vals, fmt='blue', yerr=wc_std_vals, label='WC')
plt.xlabel('K - Value')
plt.ylabel('WC-SSD')

plt.legend()
plt.show()
plt.errorbar(K_vals, sc_mean_vals, fmt='red', yerr=sc_std_vals, label='SC')
plt.xlabel('K - Value')
plt.ylabel('SC')

plt.legend()
plt.show()

wc_mean_vals = []
sc_mean_vals = []
wc_std_vals = []
sc_std_vals = []
for i in K_vals:
    wc_mean_vals.append(results[2][i]['wc'][0])
    wc_std_vals.append(results[2][i]['wc'][1])
    sc_mean_vals.append(results[2][i]['sc'][0])
    sc_std_vals.append(results[2][i]['sc'][1])

plt.errorbar(K_vals, wc_mean_vals, fmt='blue', yerr=wc_std_vals, label='WC')
plt.xlabel('K - Value')
plt.ylabel('WC-SSD')

plt.legend()
plt.show()
plt.errorbar(K_vals, sc_mean_vals, fmt='red', yerr=sc_std_vals, label='SC')
plt.xlabel('K - Value')
plt.ylabel('SC')

plt.legend()
plt.show()

wc_mean_vals = []
sc_mean_vals = []
wc_std_vals = []
sc_std_vals = []
for i in K_vals:
    wc_mean_vals.append(results[3][i]['wc'][0])
    wc_std_vals.append(results[3][i]['wc'][1])
    sc_mean_vals.append(results[3][i]['sc'][0])
    sc_std_vals.append(results[3][i]['sc'][1])

plt.errorbar(K_vals, wc_mean_vals, fmt='blue', yerr=wc_std_vals, label='WC')
plt.xlabel('K - Value')
plt.ylabel('WC-SSD')

plt.legend()
plt.show()
plt.errorbar(K_vals, sc_mean_vals, fmt='red', yerr=sc_std_vals, label='SC')
plt.xlabel('K - Value')
plt.ylabel('SC')

plt.legend()
plt.show()
"""
