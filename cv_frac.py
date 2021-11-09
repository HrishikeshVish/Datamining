import pandas as pd
import numpy as np
from trees import treeNode, calcAccuracy, calcAccuracyBagging
from statistics import mean, mode, stdev
import copy
import random
import math
import json
pd.options.mode.chained_assignment = None

trainingSet = pd.read_csv('trainingSet.csv')
trainingSet = trainingSet.sample(random_state=18, frac=1)
def bagging(train, depth):
    trees = []
    for i in range(30):
        randomState = random.randint(1, 50)
        pseudoSample = trainingSet.sample(random_state=randomState, frac=0.5)
        obj = treeNode(0, children={0:0, 1:0})
        tree = obj.buildTree(train, 0, 8)
        trees.append(tree)
    return trees
def randomForests(trainingSet, depth):
    trees = []
    for i in range(30):
        
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
    fracs = [0.05, 0.075, 0.1, 0.15, 0.2]
    depthMap = {}
    for j in range(len(fracs)):
        trains = []
        tests = []
        for i in range(10):
            testSet = train_slices[i]
            new_train = copy.copy(train_slices)
            new_train.pop(i)
            new_train = pd.concat(new_train)
            new_train.reset_index(drop = True, inplace=True)
            new_train = new_train.sample(random_state=32, frac = fracs[j])
            new_train.reset_index(drop=True, inplace=True)
            if(model == 'dt'):
                obj = treeNode(0, children={0:0, 1:0}, maxDepth=8)
                tree = obj.buildTree(new_train, 0, 8)
            
                train_acc, test_acc = calcAccuracy(tree, new_train, testSet)
            if(model == 'bt'):
                tree = bagging(new_train, 8)
                test_acc = calcAccuracyBagging(tree, new_train, testSet)
                train_acc = 0
            if(model == 'rf'):
                tree = randomForests(new_train, 8)
                test_acc = calcAccuracyBagging(tree, new_train, testSet)
                train_acc = 0
            trains.append(train_acc)
            tests.append(test_acc)
        depthMap[str(fracs[j])] = [mean(trains), mean(tests), stdev(tests)/math.sqrt(10)]
    print(depthMap)
            
        
    return 0
#bag = crossValidation('bt', trainingSet)
#det = crossValidation('dt', trainingSet)
raf = crossValidation('rf', trainingSet)
#with open('bag_frac.json', 'w') as fp:
#    json.dump(bag, fp)
with open('rf_frac.json', 'w') as fp:
    json.dump(raf, fp)
#with open('det_frac.json', 'w') as fp:
#    json.dump(det, fp)
