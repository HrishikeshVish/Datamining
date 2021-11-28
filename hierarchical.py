import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist, cdist
from statistics import mean
import math
import copy
def getCentroids(sorted_data):
    classes = sorted_data.keys()
    centroids = {}
    i = 0
    for clas in classes:
        elements = sorted_data[clas]
        x = 0
        y = 0
        
        for element in elements:
            x += element[0]
            y += element[1]
        centroids[i] = [x/len(elements), y/len(elements)]
        i+=1
    return centroids
def euclidean_dist(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1] - point2[1])**2)

def within_cluster_ssd(sorted_data, centroids):
    classes = sorted_data.keys()
    wc_ssd = 0
    for i in classes:
        elements = sorted_data[i]
        for element in elements:
            wc_ssd += (euclidean_dist(element, centroids[i])**2)
    return wc_ssd
def silCoef(sorted_data, input_dataset):
    classes = sorted_data.keys()
    S_i = []
    for clas in classes:
        cur_cluster = set(tuple(x) for x in sorted_data[clas])
        input_set = set(tuple(x) for x in input_dataset)
        rem_cluster = []
        for x in input_set:
            if x not in cur_cluster:
                rem_cluster.append(x)
        rem_cluster = np.asarray(rem_cluster)
        cur_cluster = np.asarray(sorted_data[clas])
        self_dist = cdist(cur_cluster, cur_cluster, metric='euclidean')
        other_dist = cdist(cur_cluster, rem_cluster, metric='euclidean')
        A_sum = np.sum(self_dist, axis=1)/(len(self_dist[0]))
        B_sum = np.sum(other_dist, axis=1)/(len(other_dist[0]))
        S_i.extend((B_sum[i]-A_sum[i])/max(A_sum[i], B_sum[i]) for i in range(len(A_sum)))
    return mean(S_i)
               
def nmi(sorted_data, input_dataset):
    classes = sorted_data.keys()
    #print(sorted_data)
    p_c = {}
    for i in range(10):
        p_c[i] = 10/len(input_dataset)
    prob_classes = 0
    for clas in range(10):
        prob_classes += p_c[clas] * math.log(p_c[clas], math.e)
    prob_cluster = 0
    for cluster in classes:
        prob_cluster += (len(sorted_data[cluster])/len(input_dataset)) * math.log((len(sorted_data[cluster])/len(input_dataset)), math.e)
        
    inf_gain = 0
    for cluster in classes:
        ground_class = [i[0] for i in sorted_data[cluster]]
        info_gain_cluster = 0
        for clas in set(ground_class):
            expansion = ground_class.count(clas)/len(ground_class)
            info_gain_cluster+= expansion *math.log(expansion, math.e)
        info_gain_cluster = -1 * (len(sorted_data[cluster])/len(input_dataset)) * info_gain_cluster
        inf_gain += info_gain_cluster
    inf_gain = (-1*prob_classes) - inf_gain
            
    
    nmi = inf_gain / (-1*(prob_cluster + prob_classes))
    return nmi
        

dataset1 = pd.read_csv('digits-embedding.csv', header=None)
dataset1.columns = ['id', 'class', 'feature1', 'feature2']
class0 = dataset1[dataset1['class'] == 0].sample(n=10, random_state=6)
class1 = dataset1[dataset1['class'] == 1].sample(n=10, random_state=6)
class2 = dataset1[dataset1['class'] == 2].sample(n=10, random_state=6)
class3 = dataset1[dataset1['class'] == 3].sample(n=10, random_state=6)
class4 = dataset1[dataset1['class'] == 4].sample(n=10, random_state=6)
class5 = dataset1[dataset1['class'] == 5].sample(n=10, random_state=6)
class6 = dataset1[dataset1['class'] == 6].sample(n=10, random_state=6)
class7 = dataset1[dataset1['class'] == 7].sample(n=10, random_state=6)
class8 = dataset1[dataset1['class'] == 8].sample(n=10, random_state=6)
class9 = dataset1[dataset1['class'] == 9].sample(n=10, random_state=6)

input_dataset = pd.concat([class0, class1, class2, class3, class4, class5,
                           class6, class7, class8, class9])
input_dataset = input_dataset.sample(frac=1, random_state=24)
input_dataset = input_dataset.drop('id', axis=1)
class_labels = list(input_dataset['class'])
nmi_input_dataset = np.asarray(input_dataset)
input_dataset = input_dataset.drop('class', axis=1)

cluster_single = linkage(input_dataset, method='single', metric='euclidean')
cluster_complete = linkage(input_dataset, method='complete', metric='euclidean')
cluster_average = linkage(input_dataset, method='average', metric='euclidean')
#cut_clusters = cut_tree(cluster, n_clusters=2)
"""
plt.figure(figsize=(25, 10))
plt.title("Agglomerative Clustering - Single Linkage")
plt.xlabel("Data Point")
plt.ylabel("Distance")
dendrogram(cluster_single, leaf_rotation=90, leaf_font_size=8, labels=class_labels)
plt.show()


plt.figure(figsize=(25, 10))
plt.title("Agglomerative Clustering - Complete Linkage")
plt.xlabel("Data Point")
plt.ylabel("Distance")
dendrogram(cluster_complete, leaf_rotation=90, leaf_font_size=8, labels=class_labels)
plt.show()


plt.figure(figsize=(25, 10))
plt.title("Agglomerative Clustering - Average Linkage")
plt.xlabel("Data Point")
plt.ylabel("Distance")
dendrogram(cluster_average, leaf_rotation=90, leaf_font_size=8, labels=class_labels)
plt.show()
"""
cut_cluster_single = cut_tree(cluster_single, n_clusters=[2,4,8,16,32]).T
cut_cluster_complete = cut_tree(cluster_complete, n_clusters=[2,4,8,16,32]).T
cut_cluster_average = cut_tree(cluster_average, n_clusters=[2,4,8,16,32]).T
input_dataset = np.asarray(input_dataset)
res_dict = {'single':{2:[], 4:[], 8:[], 16:[], 32:[]}, 'complete':{2:[], 4:[], 8:[], 16:[], 32:[]}, 'average':{2:[], 4:[], 8:[], 16:[], 32:[]}}
count = 2
for cluster_K in cut_cluster_single:
    ele_dict = {}
    ele_dict_nmi = {}
    for element in range(len(cluster_K)):
        if(cluster_K[element] not in ele_dict.keys()):
            ele_dict[cluster_K[element]] = []
            ele_dict_nmi[cluster_K[element]] = []
        ele_dict[cluster_K[element]].append(list(input_dataset[element]))
        ele_dict_nmi[cluster_K[element]].append(list(nmi_input_dataset[element]))
    centroids = getCentroids(ele_dict)
    wcssd = within_cluster_ssd(ele_dict, centroids)
    print("NMI = ", nmi(ele_dict_nmi, input_dataset))
    res_dict['single'][count].append(wcssd)
    res_dict['single'][count].append(silCoef(ele_dict, input_dataset))
    count*=2
count = 2
for cluster_K in cut_cluster_complete:
    ele_dict = {}
    ele_dict_nmi = {}
    for element in range(len(cluster_K)):
        if(cluster_K[element] not in ele_dict.keys()):
            ele_dict[cluster_K[element]] = []
            ele_dict_nmi[cluster_K[element]] = []
        ele_dict[cluster_K[element]].append(list(input_dataset[element]))
        ele_dict_nmi[cluster_K[element]].append(list(nmi_input_dataset[element]))
    centroids = getCentroids(ele_dict)
    wcssd = within_cluster_ssd(ele_dict, centroids)
    print("NMI = ", nmi(ele_dict_nmi, input_dataset))
    res_dict['complete'][count].append(wcssd)
    res_dict['complete'][count].append(silCoef(ele_dict, input_dataset))
    count*=2
count = 2
for cluster_K in cut_cluster_average:
    ele_dict = {}
    ele_dict_nmi = {}
    for element in range(len(cluster_K)):
        if(cluster_K[element] not in ele_dict.keys()):
            ele_dict[cluster_K[element]] = []
            ele_dict_nmi[cluster_K[element]] = []
        ele_dict[cluster_K[element]].append(list(input_dataset[element]))
        ele_dict_nmi[cluster_K[element]].append(list(nmi_input_dataset[element]))
    centroids = getCentroids(ele_dict)
    wcssd = within_cluster_ssd(ele_dict, centroids)
    print("NMI = ", nmi(ele_dict_nmi, input_dataset))
    res_dict['average'][count].append(wcssd)
    res_dict['average'][count].append(silCoef(ele_dict, input_dataset))
    count*=2

wc = []
silC = []
for K in [2,4,8,16,32]:
    wc.append(res_dict['single'][K][0])
    silC.append(res_dict['single'][K][1])
plt.plot([2,4,8,16,32], wc)
plt.xlabel("K Values")
plt.ylabel("WCSSD Values")
plt.title("K vs WCSSD in Single")
plt.show()

plt.plot([2,4,8,16,32], silC)
plt.xlabel("K Values")
plt.ylabel("silCoef Values")
plt.title("K vs sil coef in Single")
plt.show()

wc = []
silC = []
for K in [2,4,8,16,32]:
    wc.append(res_dict['complete'][K][0])
    silC.append(res_dict['complete'][K][1])
plt.plot([2,4,8,16,32], wc)
plt.xlabel("K Values")
plt.ylabel("WCSSD Values")
plt.title("K vs WCSSD in Complete")
plt.show()

plt.plot([2,4,8,16,32], silC)
plt.xlabel("K Values")
plt.ylabel("silCoef Values")
plt.title("K vs sil coef in Complete")
plt.show()


wc = []
silC = []
for K in [2,4,8,16,32]:
    wc.append(res_dict['average'][K][0])
    silC.append(res_dict['average'][K][1])
plt.plot([2,4,8,16,32], wc)
plt.xlabel("K Values")
plt.ylabel("WCSSD Values")
plt.title("K vs WCSSD in Average")
plt.show()

plt.plot([2,4,8,16,32], silC)
plt.xlabel("K Values")
plt.ylabel("silCoef Values")
plt.title("K vs sil coef in Average")
plt.show()




    
