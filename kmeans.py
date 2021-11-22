import pandas as pd
import numpy as np
from statistics import mode, mean
import copy
import math
from scipy.spatial.distance import cdist
import time
np.random.seed(0)

data = pd.read_csv('digits-embedding.csv', header=None)
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
        #membershipVector[i].append(list(init_centroids[i]))
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
def calcStuff(membershipVector, init_centroids, data, membershipDict, distances=[]):
    wc_ssd = 0
    silCoef = 0
    nmi = 0
    for i in range(len(membershipVector)):
        for ele in membershipVector[i]:
            wc_ssd += euclidean_dist(ele, init_centroids[i])
    

    
    classes = list(data['class'])
    p_c = {}
    for i in set(classes):
        p_c[i] = classes.count(i)/len(classes)
    inf_gain = 0
    
    for cluster in membershipVector:
        cur_classes = [i[1] for i in cluster]
        for clas in set(cur_classes):
            
            expansion = cur_classes.count(clas)/len(cur_classes)
            inf_gain += expansion * math.log(expansion/(p_c[clas] *(len(cluster)/len(data))))
    prob_classes = 0
    for clas in set(classes):
        prob_classes += p_c[clas] * math.log(p_c[clas])
    prob_cluster = 0
    for cluster in membershipVector:
        prob_cluster += (len(cluster)/len(data)) *math.log((len(cluster)/len(data)))
    nmi = inf_gain / (-1*(prob_cluster + prob_classes))
        
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


    del new_data
    clusters = list(set(membershipDict.values()))
    S_i = []
    count = 1
    for cluster in clusters:
        elements = [int(i) for i, v in membershipDict.items() if v == cluster]
        other_elements = [int(i) for i,v in membershipDict.items() if v!=cluster]



        cluster_dist = np.take(distances, elements, axis=0)
        values = list(membershipDict.values())
        cluster2 = pd.DataFrame(cluster_dist)

        A_list = np.asarray(cluster2.drop(other_elements, axis=1))
        B_list = np.asarray(cluster2.drop(elements, axis=1))
        del cluster2

        #A_list = np.take(cluster_dist.T, elements, axis=0).T
        #exit()
        #B_list = np.take(cluster_dist.T, other_elements, axis=0).T

        A_sum = np.sum(A_list, axis=1)/(len(A_list[0])-1)
        
        B_sum = np.sum(B_list, axis=1)/(len(B_list[0]))
        del A_list
        del B_list
        count = 0


        S_i.extend([((B_sum[i]-A_sum[i])/max(A_sum[i], B_sum[i])) for i in range(len(A_sum))])

        """
        for element in elements:

            A_list = np.multiply(cur_cluster, distances[element])
            B_list = np.multiply(other_elem, distances[element])

            A_val = np.sum(A_list)
            A_val = A_val/(len(A_list)-1)
            B_val = np.sum(B_list)
            B_val = B_val/len(B_list)
            S_i.append((B_val-A_val)/max(B_val, A_val))
            print(count)
            count+=1
        """
    silCoef = mean(S_i)

         
            
    return wc_ssd, silCoef, nmi
def kmeans(K, data, distances_valid=[]):
    data.columns = ['id', 'class', 'feature1', 'feature2']
    initial_centroid_indices = np.random.choice(len(data), K, replace=False)
    init_centroids = data.iloc[initial_centroid_indices]
    #print(init_centroids)
    indexes = list(init_centroids.index)
    #remaining_points = data.drop(indexes)
    remaining_points = data
    init_centroids = np.asarray(init_centroids)
    rem_points = copy.copy(remaining_points)
    rem_points.set_index(['id', 'class'])
    rem_points = rem_points.drop('id', axis=1)
    rem_points = rem_points.drop('class', axis=1)
    remaining_points = np.asarray(remaining_points)
    membershipDict = {}
    membershipVector = []
    for i in range(K):
        membershipVector.append([])
    vector_euclid_dist = np.vectorize(euclidean_dist, signature='(n),(m)->()')
    init_centroids = pd.DataFrame(init_centroids)
    init_centroids.columns = ['id', 'class', 'x', 'y']
    init_centroids = init_centroids.drop('id', axis=1)
    init_centroids = init_centroids.drop('class', axis=1)
    init_centroids = np.asarray(init_centroids)
    distances = cdist(rem_points, init_centroids, metric='euclidean')
    

    for i in range(len(distances)):
        min_dist_index = list(distances[i]).index(min(distances[i]))
        membershipDict[remaining_points[i][0]] = min_dist_index
        membershipVector[min_dist_index].append(list(remaining_points[i]))
    #print(init_centroids)
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
        distances = cdist(rem_points, init_centroids, metric='euclidean')
        cur_indices = np.asarray(list(membershipDict.values()))
        min_dist_indices = np.asarray(distances.argmin(axis=1))        
        changed_indices = np.where(cur_indices != min_dist_indices, 1, 0)
        min_dist_index = [value for value in np.where(cur_indices != min_dist_indices, min_dist_indices, -1) if value !=-1]

        indices = list(np.where(changed_indices == 1)[0])

        for i in range(len(indices)):
            cur_index = membershipDict[remaining_points[indices[i]][0]]
            membershipVector[cur_index].remove(list(remaining_points[indices[i]]))
            membershipVector[min_dist_index[i]].append(list(remaining_points[indices[i]]))
            membershipDict[remaining_points[indices[i]][0]] = min_dist_index[i]
            changed_any = True
            count+=1

        if(epoch == 48):
            break
        if(changed_any == False):
            changed = False
        
        
        else:
            print("Epoch ", str(epoch), " ", str(count), " Element(s) Changed")

            init_centroids = change_centroid(init_centroids, membershipVector)


    del rem_points
    wc_ssd, silCoef, nmi = calcStuff(membershipVector, init_centroids, data, membershipDict, distances_valid)
    print("WC-SSD: ",str(wc_ssd))
    print("SC: ", str(silCoef))
    print("NMI: ", str(nmi))
    #print(membershipVector)
    return wc_ssd, silCoef, nmi

if __name__ == '__main__':
    kmeans(10, data)    
