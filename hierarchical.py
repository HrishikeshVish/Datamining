import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
dataset1 = pd.read_csv('digits-embedding.csv', header=None)
dataset1.columns = ['id', 'class', 'feature1', 'feature2']
dataset1 = dataset1.drop('id', axis=1)
dataset1 = dataset1.drop('class', axis=1)
#plt.scatter(dataset1['feature1'], dataset1['feature2'])
#plt.show()
cluster = linkage(dataset1, method='single', metric='euclidean')
print(cluster)
plt.figure(figsize=(25, 10))
plt.title("Agglomerative Clustering")
plt.xlabel("Data Point")
plt.ylabel("Distance")
dendrogram(cluster, leaf_rotation=90, leaf_font_size=8,)
plt.savefig('dendrogram_single.png')
