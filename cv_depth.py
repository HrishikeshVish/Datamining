import pandas as pd
import numpy as np
from trees import treeNode, calcAccuracy, calcAccuracyBagging
from statistics import mean, mode, stdev
import copy
import random
import json
import math
pd.options.mode.chained_assignment = None

trainingSet = pd.read_csv('trainingSet.csv')
testSet = pd.read_csv('testSet.csv')
trainingSet = trainingSet.sample(random_state=18, frac=1)
trainingSet = trainingSet.sample(random_state=32, frac=0.5)

def bagging(train, depth):
    trees = []
    for i in range(30):
        randomState = random.randint(1, 50)
        pseudoSample = trainingSet.sample(random_state=randomState, frac=1)
        obj = treeNode(0, children={0:0, 1:0})
        tree = obj.buildTree(train, 0, depth)
        trees.append(tree)
    return trees
def randomForests(trainingSet, depth):
    trees = []
    for i in range(30):
        
        randomState = random.randint(1, 50)
        pseudoSample = trainingSet.sample(random_state=randomState, frac=1, replace=True)
        obj = treeNode(0, children={0:0, 1:0})
        tree = obj.buildTree(trainingSet, 0, depth, rf=True, columns=list(pseudoSample.columns), p=7)
        trees.append(tree)
        #printTreeRecurse(tree,0)
    #calcAccuracyBagging(trees, trainingSet, testSet)
    return trees
    
def crossValidation(model, trainingSet):
    train_slices = []
    for i in range(0, len(trainingSet), 260):
        train_slices.append(trainingSet[i:i+260])
    depth = [3,5,7,9]
    depthMap = {}
    for j in depth:
        trains = []
        tests = []
        for i in range(10):
            testSet = train_slices[i]
            new_train = copy.copy(train_slices)
            new_train.pop(i)
            new_train = pd.concat(new_train)
            new_train.reset_index(drop = True, inplace=True)
            if(model == 'dt'):
                obj = treeNode(0, children={0:0, 1:0}, maxDepth=j)
                tree = obj.buildTree(new_train, 0, j)
            
                train_acc, test_acc = calcAccuracy(tree, new_train, testSet)
            if(model == 'bt'):
                tree = bagging(new_train, j)
                test_acc = calcAccuracyBagging(tree, new_train, testSet)
                train_acc = 0
            if(model == 'rf'):
                tree = randomForests(new_train, j)
                test_acc = calcAccuracyBagging(tree, new_train, testSet)
                train_acc=0
            trains.append(train_acc)
            tests.append(test_acc)
        depthMap[j] = [mean(trains), mean(tests), stdev(tests)/math.sqrt(10)]
    print(depthMap)
         
        
    return 0
raf = crossValidation('rf', trainingSet)
det = crossValidation('dt', trainingSet)
bag = crossValidation('bt', trainingSet)
with open('bag_depth.json', 'w') as fp:
    json.dump(bag, fp)
with open('rf_depth.json', 'w') as fp:
    json.dump(raf, fp)
with open('det_depth.json', 'w') as fp:
    json.dump(det, fp)
