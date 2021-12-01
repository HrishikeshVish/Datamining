import pandas as pd
import numpy as np
from kmeans import kmeans
import matplotlib.pyplot as plt
import random
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
wc, sc, nmi, membershipDict1 = kmeans(16, dataset1, retCluster=True)
wc, sc, nmi, membershipDict2 = kmeans(4, dataset2, retCluster=True)
wc, sc, nmi, membershipDict3 = kmeans(2, dataset3, retCluster=True)
sample1 = dataset1.sample(n=1000, random_state=23)
sample2 = dataset2.sample(n=1000, random_state=23)
sample3 = dataset3.sample(n=1000, random_state=23)

indexes = list(sample1.index)
indexes2 = list(sample2.index)
indexes3 = list(sample3.index)

for i in indexes:
    sample1['class'][i] = membershipDict1[sample1['id'][i]]
for i in indexes2:
    sample2['class'][i] = membershipDict2[sample2['id'][i]]

for i in indexes3:
    sample3['class'][i] = membershipDict3[sample3['id'][i]]

color16 = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         for i in range(16)]
color8 = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         for i in range(8)]
color4 = ['#'+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         for i in range(4)]
colors1 = []
class1 = sample1['class']
for i in class1:
    colors1.append(color16[i])
    
colors2 = []
class2 = sample2['class']
for i in class2:
    colors2.append(color8[i])
    
colors3 = []
class3 = sample3['class']
for i in class3:
    colors3.append(color4[i])

x_coord1 = sample1['feature1']
y_coord1 = sample1['feature2']
x_coord2 = sample2['feature1']
y_coord2 = sample2['feature2']
x_coord3 = sample3['feature1']
y_coord3 = sample3['feature2']

plt.scatter(x_coord1, y_coord1, color=colors1)
plt.show()

plt.scatter(x_coord2, y_coord2, color=colors2)
plt.show()

plt.scatter(x_coord3, y_coord3, color=colors3)
plt.show()
