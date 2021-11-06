import pandas as pd
import numpy as np
from trees import treeNode, calcAccuracy, calcAccuracyBagging
from statistics import mean, mode
import copy
import random
pd.options.mode.chained_assignment = None

trainingSet = pd.read_csv('trainingSet.csv')
testSet = pd.read_csv('testSet.csv')
trainingSet = trainingSet.sample(random_state=18, frac=1)
trainingSet = trainingSet.sample(random_state=32, frac=0.5)

def bagging(train, depth):
    trees = []
    for i in range(30):
        randomState = random.randint(1, 50)
        pseudoSample = trainingSet.sample(random_state=randomState, frac=0.5)
        obj = treeNode(0, children={0:0, 1:0})
        tree = obj.buildTree(train, 0, depth)
        trees.append(tree)
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
                test_acc = calcAccuracyBagging(tree, testSet)
                train_acc = 0
            trains.append(train_acc)
            tests.append(test_acc)
        depthMap[j] = [mean(trains), mean(tests)]
    #print(depthMap)
            
        
    return 0
crossValidation('bt', trainingSet)
