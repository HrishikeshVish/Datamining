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

def bagging(train, numTrees):
    trees = []
    for i in range(numTrees):
        randomState = random.randint(1, 50)
        pseudoSample = trainingSet.sample(random_state=randomState, frac=1, replace=True)
        obj = treeNode(0, children={0:0, 1:0})
        tree = obj.buildTree(train, 0, 8)
        trees.append(tree)
    return trees
def randomForests(trainingSet, numTrees):
    trees = []
    for i in range(numTrees):
        
        randomState = random.randint(1, 50)
        pseudoSample = trainingSet.sample(random_state=randomState, frac=1, replace=True)
        obj = treeNode(0, children={0:0, 1:0})
        tree = obj.buildTree(trainingSet, 0, 8, rf=True, columns=list(pseudoSample.columns), p=7)
        trees.append(tree)
        #printTreeRecurse(tree,0)
    #calcAccuracyBagging(trees, trainingSet, testSet)
    return trees
    
def crossValidation(model, trainingSet):
    train_slices = []
    for i in range(0, len(trainingSet), 260):
        train_slices.append(trainingSet[i:i+260])
    numTrees = [10, 20, 40, 50]
    depthMap = {}
    for j in numTrees:
        trains = []
        tests = []
        for i in range(10):
            testSet = train_slices[i]
            new_train = copy.copy(train_slices)
            new_train.pop(i)
            new_train = pd.concat(new_train)
            new_train.reset_index(drop = True, inplace=True)
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
bag = crossValidation('bt', trainingSet)
rf = crossValidation('rf', trainingSet)
with open('bag_num_tree.json', 'w') as fp:
    json.dump(bag, fp)
with open('rf_num_tree.json', 'w') as fp:
    json.dump(rf, fp)
