import pandas as pd
import numpy as np
from statistics import mode, mean
import copy
import math
from scipy.spatial.distance import cdist
import time
import argparse

pd.options.mode.chained_assignment = None
np.seterr(all='ignore')


def euclidean_dist(point1, point2):
    dist = 0.0
    for i in range(2, len(point1)):
        dist += np.square(point1[i] - point2[i-2])
    return np.sqrt(dist)
def find_ele(element, vector):
    for i in range(len(vector)):
        if element in vector[i]:
            return i
    return -1
def change_centroid(init_centroids, membershipVector):
    for i in range(len(init_centroids)):
        coord1 = [j[2] for j in membershipVector[i]]
        coord2 = [j[3] for j in membershipVector[i]]
        init_centroids[i][0] = mean(coord1)
        init_centroids[i][1] = mean(coord2)
    return init_centroids
def dist_in_vector(ele, curVector):
    dist = 0.0
    for i in curVector:
        dist+= euclidean_dist(ele, i)
    return dist/(len(curVector)-1)
def wcssd(membershipVector, init_centroids):
    wc_ssd = 0
    for i in range(len(membershipVector)):
        for ele in membershipVector[i]:
            wc_ssd += (euclidean_dist(ele, init_centroids[i])**2)
    return wc_ssd

def calcNMI(data, membershipVector):
    nmi = 0
    
    classes = list(data['class'])
    p_c = {}
    for i in sorted(list(set(classes))):
        p_c[i] = classes.count(i)/len(classes)
    inf_gain = 0
    p_g = {}
    for i in range(len(membershipVector)):
        p_g[i] = len(membershipVector[i])/len(classes)
    inf_gain  = 0
    prob_classes = 0
    for clas in sorted(list(set(classes))):
        prob_classes += p_c[clas] * math.log(p_c[clas], math.e)
    prob_cluster = 0

    for cluster in membershipVector:
        p_cluster = len(cluster)/len(classes)
        prob_cluster += p_cluster *math.log(p_cluster ,math.e)
    for cluster in range(len(membershipVector)):
        cur_classes = [i[1] for i in membershipVector[cluster]]
        inf_gain_cluster = 0
        for clas in set(cur_classes):
            
            expansion = cur_classes.count(clas)/len(cur_classes)
            #inf_gain += expansion * math.log(expansion/(p_c[clas] *p_g[cluster]),2)
            inf_gain_cluster += expansion * math.log(expansion,math.e)
        inf_gain_cluster = -1 * p_g[cluster] * inf_gain_cluster
        inf_gain += inf_gain_cluster
    inf_gain = (-1*prob_classes) - inf_gain

    nmi = inf_gain / (-1*(prob_cluster + prob_classes))
    return nmi
def calcStuff(membershipVector, init_centroids, data, membershipDict, distances=[], membershipVector_ind =[]):
    
    wc_ssd = wcssd(membershipVector, init_centroids)
    nmi = calcNMI(data, membershipVector)
    silCoef = 0

    S_i = []
    count = 0
    for i in range(len(membershipVector)):
        membershipVector[i]  = pd.DataFrame(membershipVector[i])
        membershipVector[i].columns = ['id', 'class', 'fea1', 'fea2']
        membershipVector[i] = membershipVector[i].drop('id', axis=1)
        membershipVector[i] = membershipVector[i].drop('class', axis=1)
        membershipVector[i] = np.asarray(membershipVector[i])
    start = time.time()
    new_data = copy.copy(data)
    new_data = new_data.drop('id', axis=1)
    new_data = new_data.drop('class', axis=1)
    if(len(distances) == 0):
        distances = cdist(new_data, new_data, metric='euclidean')

    A_sum = []
    pair_distances_other = {}
    start = time.time()
    for i in range(len(membershipVector)):
        if(i not in pair_distances_other.keys()):
            pair_distances_other[i] = []
        for j in range(i, len(membershipVector)):
            distance = cdist(membershipVector[i], membershipVector[j], metric='euclidean')
            if(i == j):
                A_sum.extend(np.sum(distance, axis=1)/(len(membershipVector[i])-1))
            else:
                if j not in pair_distances_other.keys():
                    pair_distances_other[j] = []
                pair_distances_other[i].append(np.sum(distance,axis=1)/len(membershipVector[j]))
                pair_distances_other[j].append(np.sum(distance.T, axis=1)/len(membershipVector[i]))
    end = time.time()
    B_sum = []
    for keys in sorted(list(pair_distances_other.keys())):
        pair_distances_other[keys] = np.asarray(pair_distances_other[keys])
        B_sum.extend(np.min(pair_distances_other[keys], axis=0))
    S_i.extend([((B_sum[i]-A_sum[i])/max(A_sum[i], B_sum[i])) for i in range(len(A_sum))])

    silCoef = mean(S_i)

         
            
    return wc_ssd, silCoef, nmi
def kmeans(K, data, distances_valid=[], retCluster=False, seed_value = 0, print_epoch=False):
    data.columns = ['id', 'class', 'feature1', 'feature2']
    np.random.seed(seed_value)
    initial_centroid_indices = np.random.choice(len(data), K, replace=False)
    init_centroids = data.iloc[initial_centroid_indices]
    init_centroids = init_centroids.drop('id', axis=1)
    init_centroids = init_centroids.drop('class', axis=1)
    init_centroids = np.asarray(init_centroids)
    
    remaining_points = np.asarray(data)
    
    membershipDict = {}
    membershipVector = []
    membershipVector_ind = []
    for i in range(K):
        membershipVector.append([])
        membershipVector_ind.append([])

    distances = cdist(data[data.columns[[2,3]]], init_centroids, metric='euclidean')
    

    for i in range(len(distances)):
        min_dist_index = list(distances[i]).index(min(distances[i]))
        membershipDict[int(remaining_points[i][0])] = min_dist_index
        membershipVector[min_dist_index].append(list(remaining_points[i]))
    for i in range(len(init_centroids)):
        coord1 = [j[2] for j in membershipVector[i]]
        coord2 = [j[3] for j in membershipVector[i]]
        init_centroids[i][0] = mean(coord1)
        init_centroids[i][1] = mean(coord2)

    changed = True
    for epoch in range(49):
        if changed == False:
            break
        changed_any = False
        count = 0
        start = time.time()
        distances = cdist(data[data.columns[[2,3]]], init_centroids, metric='euclidean')
        cur_indices = np.asarray(list(membershipDict.values()))
        min_dist_indices = np.asarray(distances.argmin(axis=1))        
        changed_indices = np.where(cur_indices != min_dist_indices, 1, 0)
        min_dist_index = [value for value in np.where(cur_indices != min_dist_indices, min_dist_indices, -1) if value !=-1]

        indices = list(np.where(changed_indices == 1)[0])

        for i in range(len(indices)):
            cur_index = membershipDict[remaining_points[indices[i]][0]]
            membershipVector[cur_index].remove(list(remaining_points[indices[i]]))
            
            membershipVector[min_dist_index[i]].append(list(remaining_points[indices[i]]))
            
            membershipDict[int(remaining_points[indices[i]][0])] = min_dist_index[i]
            changed_any = True
            count+=1

        if(epoch == 48):
            for key, value in membershipDict.items():
                membershipVector_ind[value].append(key)
            break
        if(changed_any == False):
            for key, value in membershipDict.items():
                membershipVector_ind[value].append(key)
            changed = False
        
        
        else:
            if(print_epoch):
                print("Epoch ", str(epoch), " ", str(count), " Element(s) Changed")

            init_centroids = change_centroid(init_centroids, membershipVector)


    wc_ssd, silCoef, nmi = calcStuff(membershipVector, init_centroids, data, membershipDict, distances_valid, membershipVector_ind)
    print("WC-SSD: ",str(wc_ssd))
    print("SC: ", str(silCoef))
    print("NMI: ", str(nmi))

    if(retCluster == True):
        return wc_ssd, silCoef, nmi, membershipDict
    return wc_ssd, silCoef, nmi
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataFileName')
    parser.add_argument('kvalue')
    args = parser.parse_args()
    data = args.dataFileName
    k_value = args.kvalue
    data = pd.read_csv(data, header=None)
    kmeans(int(k_value), data)    
