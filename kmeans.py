import pandas as pd
import numpy as np
from statistics import mode, mean
import copy
import math
from scipy.spatial.distance import cdist
np.random.seed(0)

data = pd.read_csv('digits-embedding.csv')
def euclidean_dist(point1, point2):
    dist = 0.0
    for i in range(2, len(point1)):
        dist += ((point1[i] - point2[i])**2)
    return dist**0.5
def find_ele(element, vector):
    for i in range(len(vector)):
        if element in vector[i]:
            return i
    return -1
def change_centroid(init_centroids, membershipVector):
    for i in range(len(init_centroids)):
        membershipVector[i].append(list(init_centroids[i]))
        coord1 = [j[2] for j in membershipVector[i]]
        coord2 = [j[3] for j in membershipVector[i]]
        init_centroids[i][2] = mean(coord1)
        init_centroids[i][3] = mean(coord2)
    return init_centroids
def dist_in_vector(ele, curVector):
    dist = 0.0
    for i in curVector:
        dist+= euclidean_dist(ele, i)
    return dist/(len(curVector)-1)
def calcStuff(membershipVector, init_centroids, data):
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
    
    for i in range(len(membershipVector)):
        curVector = membershipVector[i]
        remVectors = copy.copy(membershipVector)

        #print(remVectors)
        #print(curVector)
        
        try:
            remVectors.remove(curVector)
            remVectors = [item for sublist in remVectors for item in sublist]
        except:
            remVectors = [list(item) for sublist in remVectors for item in sublist]
            for item in curVector:
                if item in remVectors:
                    remVectors.remove(item)
        
        #print(remVectors)
        #print(len(remVectors))
        distances_self = cdist(curVector, curVector)
        A_vals = [sum(i)/(len(i)-1) for i in distances_self]
        
        distances_other = cdist(curVector, remVectors)
        B_vals = [sum(i)/len(i) for i in distances_other]
        for i in range(len(B_vals)):
            S_i.append((B_vals[i] - A_vals[i])/max(A_vals[i], B_vals[i]))
    silCoef = mean(S_i)            
         
            
    return wc_ssd, silCoef, nmi
def kmeans(K, data):
    data.columns = ['id', 'class', 'feature1', 'feature2']
    initial_centroid_indices = np.random.choice(len(data), K, replace=False)
    init_centroids = data.iloc[initial_centroid_indices]
    #print(init_centroids)
    indexes = list(init_centroids.index)
    remaining_points = data.drop(indexes)
    init_centroids = np.asarray(init_centroids)
    print(init_centroids)
    remaining_points = np.asarray(remaining_points)
    
    membershipVector = []
    for i in range(K):
        membershipVector.append([])
    vector_euclid_dist = np.vectorize(euclidean_dist, signature='(n),(m)->()')
    distances = vector_euclid_dist(remaining_points[:,np.newaxis], init_centroids)
    for i in range(len(distances)):
        min_dist_index = list(distances[i]).index(min(distances[i]))
        membershipVector[min_dist_index].append(list(remaining_points[i]))
    #print(init_centroids)
    for i in range(len(init_centroids)):
        membershipVector[i].append(list(init_centroids[i]))
        coord1 = [j[2] for j in membershipVector[i]]
        coord2 = [j[3] for j in membershipVector[i]]
        init_centroids[i][2] = mean(coord1)
        init_centroids[i][3] = mean(coord2)
    #print(init_centroids)
    changed = True
    for epoch in range(49):
        if changed == False:
            break
        changed_any = False
        count = 0
        distances = vector_euclid_dist(remaining_points[:, np.newaxis], init_centroids)
        for i in range(len(distances)):
            min_dist_index = list(distances[i]).index(min(distances[i]))
            
            if(list(remaining_points[i]) not in membershipVector[min_dist_index]):
                cur_index = find_ele(list(remaining_points[i]), membershipVector)
                membershipVector[cur_index].remove(list(remaining_points[i]))
                membershipVector[min_dist_index].append(list(remaining_points[i]))
                count +=1
                changed_any = True
        if(epoch == 48):
            break
        if(changed_any == False):
            changed = False
        
        
        else:
            print("Epoch ", str(epoch), " ", str(count), " Element(s) Changed")
            init_centroids = change_centroid(init_centroids, membershipVector)
            
    wc_ssd, silCoef, nmi = calcStuff(membershipVector, init_centroids, data)
    print("WC-SSD: ",str(wc_ssd))
    print("SC: ", str(silCoef))
    print("NMI: ", str(nmi))
    #print(membershipVector)
    return wc_ssd, silCoef, nmi

if __name__ == '__main__':
    kmeans(10, data)    
