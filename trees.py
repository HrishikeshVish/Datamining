import pandas as pd
import numpy as np
import operator
from statistics import mean, mode
import math
pd.options.mode.chained_assignment = None
import random
class treeNode:
    def __init__(self, depth, children, attribs=None,  value=None, cur='root', maxDepth=8):
        self.cur = cur
        self.depth = depth
        self.paths = attribs
        self.childNodes = children
        self.val = value
        self.maxDepth = maxDepth

    def Entropy(self, row):
        distinct_class = set(row)
        entropy = 0
        net = len(row)
        
        for cl in distinct_class:
            frac = (row.value_counts()[cl]/net)
            entropy -= frac * (math.log(frac)/math.log(2.0))
        return entropy
    def Gini(self, row):
        distinct_class = set(row)
        gini = 0
        net = len(row)
        #print(row.count(0))
        #print(row)
        for cl in distinct_class:
            frac = (row.count(cl)/net)
            gini += frac**2
        return 1 - gini

    def giniGain(self, res, attrib):
        attrib_class = set(attrib)
        classDict = {}
        #print(res)
        #print(attrib)
        
        for i in attrib_class:
            classDict[i] = []
        for j in range(len(attrib)):
            
            classDict[attrib[j]].append(res[j])
        entSum = 0
        #print(classDict)
        for i in attrib_class:

            entSum += (len(classDict[i])/len(res)) * self.Gini(classDict[i])
        return self.Gini(list(res)) - entSum


    def buildLayer(self, trainingData, rf=False, columns_full=[], p=0):
        columns = list(trainingData.columns)
        columns.remove('decision')
        if(len(columns) == 0):
            return 0, 0, 0
        giniGain = {}
        for col in columns:
            
            giniGain[col] = self.giniGain(list(trainingData['decision']), list(trainingData[col]))
        if(len(giniGain.keys()) == 1):
            bestCol = list(giniGain.keys())[0]
        else:
            bestCol = max(giniGain.items(), key=operator.itemgetter(1))[0]
        leftData = trainingData.loc[trainingData[bestCol] == 0]
        rightData = trainingData.loc[trainingData[bestCol] == 1]
        
        leftData.drop(bestCol, inplace=True, axis=1)
        rightData.drop(bestCol, inplace=True, axis=1)
        if(rf == True): #Random Forest Split
            total_cols = len(columns_full) - 1
            left_retain = []
            right_retain = []
            for i in range(p):
                left_retain.append(columns_full[random.randint(0, total_cols)])
                right_retain.append(columns_full[random.randint(0, total_cols)])
            cur_cols_left = list(leftData.columns)
            cur_cols_right = list(rightData.columns)
            left_retain_new = ['decision']
            right_retain_new = ['decision']
            for i in range(len(left_retain)):
                if(left_retain[i] in cur_cols_left):
                    left_retain_new.append(left_retain[i])
                if(right_retain[i] in cur_cols_right):
                    right_retain_new.append(right_retain[i])
            left_drop = list(set(cur_cols_left) - set(left_retain_new))
            right_drop = list(set(cur_cols_right) - set(right_retain_new))
            for i in left_drop:
                leftData.drop(i, inplace=True, axis=1)
            for i in right_drop:
                rightData.drop(i, inplace=True, axis=1)
            #print(leftData.columns)
            
                    
        return bestCol, leftData, rightData
    
    def buildTree(self, trainingData, depth, maxDepth, rf=False, columns=[], p=0):
        tree = treeNode(depth, children={0:0, 1:0})
        #print(depth)
        
        if(len(trainingData['decision']) == 0):
            #tree.value = list(trainingData['decision'])[0]
            return tree
        if(depth == maxDepth or len(trainingData)<50):
            try:
                output = mode(list(trainingData['decision']))
            except:
                
                output = list(trainingData['decision'])[0]
            tree.val = output
            return tree

        bestCol, leftData, rightData = tree.buildLayer(trainingData,rf, columns, p)
        if(bestCol == 0):
            return tree
        
            
        tree.cur = bestCol
        tree.attribs = [0,1]
        if(len(leftData.columns) == 1 and len(leftData['decision'])>0):
            try:
                tree.childNodes[0] = treeNode(depth+1, attribs=None, children=None, value = mode(list(leftData['decision'])), cur=bestCol)
                #tree.val = mode(list(leftData['decision']))
            except:
                tree.childNodes[0] = treeNode(depth+1, attribs=None, children=None, value = list(leftData['decision'])[0], cur=bestCol)
                #tree.val = list(leftData['decision'])[0]
            if(len(rightData.columns) == 1 and len(rightData['decision'])>0):
               try:
                   tree.childNodes[1] = treeNode(depth+1, attribs=None, children=None, value = mode(list(rightData['decision'])), cur=bestCol)
                   #tree.val = mode(list(rightData['decision']))
               except:
                   tree.childNodes[1] = treeNode(depth+1, attribs=None, children=None, value = list(rightData['decision'])[0], cur=bestCol)
                   tree.val = list(rightData['decision'])[0]
               return tree
            else:
                if(len(set(rightData['decision'])) == 1):
                    tree.childNodes[1] = treeNode(depth+1, attribs=None, children=None, value = list(rightData['decision'])[0], cur=bestCol)
                else:
                    rightTree = self.buildTree(rightData, depth+1, maxDepth, rf, columns,p)
                    tree.childNodes[1] = rightTree
                return tree
        elif(len(rightData.columns) == 1 and len(rightData['decision'])>0):
               if(len(set(leftData['decision'])) == 1):
                   tree.childNodes[0] = treeNode(depth+1, attribs=None, children=None, value = list(leftData['decision'])[0], cur=bestCol)
               else:
                   leftTree = self.buildTree(leftData, depth+1, maxDepth, rf, columns, p)
                   tree.childNodes[0] = leftTree
               try:
                   tree.childNodes[1] = treeNode(depth+1, attribs=None, children=None, value=mode(list(rightData['decision'])), cur=bestCol)
                   #tree.val = mode(list(rightData['decision']))
               except:
                   tree.childNodes[1] = treeNode(depth+1, attribs=None, children=None, value=list(rightData['decision'])[0], cur=bestCol)
                   #tree.val = list(rightData['decision'])[0]
               return tree
               
            
        if(len(set(leftData['decision'])) == 1):
            #print(leftData['decision'])
            tree.childNodes[0] = treeNode(depth+1, attribs=None, children=None, value = list(leftData['decision'])[0], cur=bestCol)
            if(len(set(rightData['decision'])) == 1):
                #print(rightData['decision'])
                tree.childNodes[1] = treeNode(depth+1, attribs=None, children=None, value = list(rightData['decision'])[0], cur=bestCol)
                return tree
            else:
                rightTree = self.buildTree(rightData, depth+1, maxDepth, rf, columns, p)
                tree.childNodes[1] = rightTree
                return tree
        elif(len(set(rightData['decision']))==1):
            #print(rightData['decision'])
            leftTree = self.buildTree(leftData, depth+1, maxDepth, rf, columns, p)
            tree.childNodes[0] = leftTree
            tree.childNodes[1] = treeNode(depth+1, attribs=None, children=None, value = list(rightData['decision'])[0], cur=bestCol)
            return tree
        else:
            leftTree = self.buildTree(leftData, depth+1, maxDepth, rf, columns, p)
            rightTree = self.buildTree(rightData, depth+1, maxDepth, rf, columns, p)
            
            tree.childNodes[0] = leftTree
            tree.childNodes[1] = rightTree
            return tree
def printTreeRecurse(tree, indent):
        if(type(tree) == int):
            return
        if(tree.val !=None):
                print(indent*'-', indent, tree.cur, tree.depth, end = '')
                print(' Val ', tree.val)
                return
        else:
                try:
                    print(indent*'-', indent, tree.cur, tree.depth, ' ', tree.childNodes[0].cur, ' ', tree.childNodes[1].cur, end = '')
                except:
                    print(indent*'-', indent, tree.cur, tree.depth, end = '')
                print()
                
                printTreeRecurse(tree.childNodes[0], indent+1)
                printTreeRecurse(tree.childNodes[1], indent+1)
def predict(row, tree):
    if(tree.val !=None):
        #print(tree.val)
        return tree.val
    #print(tree.cur)
    if(tree.cur == 'root'):
        return random.randint(0,1)
    return predict(row, tree.childNodes[row[tree.cur]])

def calcAccuracy(tree, trainingSet, testSet):
    count = 0
    for index, row in trainingSet.iterrows():
        output = predict(row, tree)
        if(output == row['decision']):
            count+=1
        #print(output, row['decision'])
    print("Training Acc: %0.2f"%(count/len(trainingSet)))
    train_acc = count/len(trainingSet)
    count = 0
    for index, row in testSet.iterrows():
        output = predict(row, tree)
        if(output == row['decision']):
            count+=1
        #print(output, row['decision'])
    print("Test Acc: %0.2f"%(count/len(testSet)))
    test_acc = count/len(testSet)
    return train_acc, test_acc
    
def calcAccuracyBagging(trees, trainingSet, testSet):
    count = 0
    for index, row in trainingSet.iterrows():
        aggregate = []

        for tree in trees:
            output = predict(row, tree)

            aggregate.append(output)
        #print("Number of trees correctly pred ", count_in, " correct ", row['decision'], " predicted ", mean(aggregate))
        try:
            output = int(math.round(mean(aggregate)))
        except:
            output = aggregate[0]
        if(output == row['decision']):
            count+=1
    print("Training Acc: %0.2f"%(count/len(trainingSet)))
    count = 0
    for index, row in testSet.iterrows():
        aggregate = []
        for tree in trees:
            output = predict(row, tree)
            aggregate.append(output)
        try:
            output = mode(aggregate)
        except:
            output = aggregate[0]
        if(output == row['decision']):
            count+=1
    print("Test Acc: %0.2f"%(count/len(testSet)))
    return count/len(testSet)
            
def decisionTree(trainingSet, testSet):
    obj = treeNode(0, children={0:0, 1:0})
    tree = obj.buildTree(trainingSet, 0, 8, rf=False, columns = list(trainingSet.columns), p=7)
    #printTreeRecurse(tree, 0)
    #print(tree.cur)
    #print(tree.childNodes)
    #tree.buildLayer(trainingSet)
    
    return tree


def bagging(trainingSet, testSet):
    trees = []
    for i in range(30):
        
        randomState = random.randint(1, 50)
        pseudoSample = trainingSet.sample(random_state=randomState, frac=1, replace=True)
        tree = decisionTree(pseudoSample, testSet)
        trees.append(tree)
    calcAccuracyBagging(trees, trainingSet, testSet)
    return trees

def randomForests(trainingSet, testSet):
    trees = []
    for i in range(30):
        
        randomState = random.randint(1, 50)
        pseudoSample = trainingSet.sample(random_state=randomState, frac=1, replace=True)
        obj = treeNode(0, children={0:0, 1:0})
        tree = obj.buildTree(trainingSet, 0, 8, rf=True, columns=list(pseudoSample.columns), p=7)
        trees.append(tree)
        printTreeRecurse(tree,0)
    calcAccuracyBagging(trees, trainingSet, testSet)
    return trees

trainingSet = pd.read_csv('trainingSet.csv')
testSet = pd.read_csv('testSet.csv')
#bagging(trainingSet, testSet)
tree = decisionTree(trainingSet, testSet)
calcAccuracy(tree, trainingSet, testSet)

