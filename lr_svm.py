import pandas as pd
import numpy as np
np.seterr(all='ignore')
from statistics import mean
import argparse
import copy
import math

from sklearn import svm, metrics
pd.options.mode.chained_assignment = None
parser = argparse.ArgumentParser(description='')
parser.add_argument('trainingDataFilename')
parser.add_argument('testDataFilename')
parser.add_argument('modelIdx')
args = parser.parse_args()
#train = pd.read_csv(args.trainingDataFilename)
#test = pd.read_csv(args.testDataFilename)
train = np.genfromtxt(args.trainingDataFilename, delimiter=',')
test = np.genfromtxt(args.testDataFilename, delimiter=',')
train = np.delete(train, (0), axis=0)
test = np.delete(test, (0), axis=0)
def lr(trainingSet, testSet):
    length = len(trainingSet[0]) -1
    w = np.array([0 for i in range(length)])
    num_epochs = 0
    tol = math.pow(math.e, -6)
    weight_dif = 1
    trainMat = np.matrix(trainingSet)
    results = trainMat[:, [len(trainingSet[0])-1]]
    results = np.squeeze(np.asarray(results))
    trainingSet = np.delete(trainingSet, len(trainingSet[0])-1, 1)

    
    testRes = np.matrix(testSet)[:,[len(testSet[0])-1]]
    testRes = np.squeeze(np.asarray(testRes))
    testSet = np.delete(testSet, len(testSet[0])-1, 1)

    while(num_epochs<500 and weight_dif>tol):
        num_epochs += 1
        res = np.matmul(trainingSet, w)
        if(res.any()<0):
            predictions = np.exp(res)/(1+np.exp(res))
        else:
            predictions = 1/(1+np.exp(-1*res))
        diff = predictions - results
        diff = np.matmul(diff , trainingSet)
        diff = diff + 0.01*w
        new_w = w - 0.01*diff
        weight_dif = np.linalg.norm(new_w-w)
        w = new_w
        #print("epochs = ", num_epochs, "Weight diff = ", weight_dif)
    fin_res = np.matmul(trainingSet, w)
    if(res.any()<0):
        predictions = np.exp(res)/(1+np.exp(res))
    else:
        predictions = 1/(1+np.exp(-1*res))
    predictions = np.round(predictions)
    test_res = np.matmul(testSet, w)
    if(test_res.any()<0):
        predict_test = np.exp(test_res)/(1+np.exp(test_res))
    else:
        predict_test = 1/(1+np.exp(-1*test_res))
    predict_test = np.round(predict_test)
    train_sum = 0
    total = len(predictions)
    for i in range(len(predictions)):
        if(predictions[i] == results[i]):
            train_sum+=1
    total_test = len(predict_test)
    testSum = 0
    for i in range(len(predict_test)):
        if(predict_test[i] == testRes[i]):
            testSum+=1
    print("Training Accuracy LR: ", train_sum/total)
    print("Testing Accuracy LR: ", testSum/total_test)
            
    """
    while(num_epochs<500 and weight_dif>tol):
        num_epochs +=1
        cur_outputs = []
        grads = [0 for i in range(len(trainingSet[0])-1)]
        new_w = copy.copy(w)
        for j in range(len(trainingSet[0])-1):
            curCol = trainMat[:, [j]]
            sum_grad = 0
            for i in range(len(trainingSet)):
                y_i =    trainingSet[i][len(trainingSet[i])-1]
                product = np.matmul(w, trainingSet[i][:len(trainingSet[i])-1])
                prediction = 1/(1+np.exp(-1*product))
                prediction = round(prediction)
                y_pred = prediction
                
                sum_grad += ((-1 * y_i + y_pred) * curCol[i])
            sum_grad += 0.01*w[j]
            w_new_j = w[j] - 0.01 * sum_grad
            new_w[j] = w_new_j
        dif = np.subtract(new_w, w)
        w = new_w
        weight_dif = np.linalg.norm(dif)
        print("epochs = ", num_epochs, "weight diff = ", weight_dif)
        #print(w)
    """
    
def svm(trainingSet, testSet):
    length = len(trainingSet[0]) -1
    w = np.array([0 for i in range(length)])
    num_epochs = 0
    tol = math.pow(math.e, -6)
    weight_dif = 1
    trainMat = np.matrix(trainingSet)
    results = trainMat[:, [len(trainingSet[0])-1]]
    results = np.squeeze(np.asarray(results))
    trainingSet = np.delete(trainingSet, len(trainingSet[0])-1, 1)
    
    for i in range(len(results)):
        if(results[i] == 0):
            results[i]= -1
    
    testRes = np.matrix(testSet)[:,[len(testSet[0])-1]]
    testRes = np.squeeze(np.asarray(testRes))
    
    for i in range(len(testRes)):
        if(testRes[i] == 0):
            testRes[i] = -1
    
    testSet = np.delete(testSet, len(testSet[0])-1, 1)
    for epoch in range(500):
        num_epochs += 1
        res = np.matmul(trainingSet, w)
        prod = np.multiply(res, results)
        for i in range(len(prod)):
            if(prod[i]<1):
                prod[i] = 1
            else:
                prod[i] = 0
        del_j = np.multiply(results, prod)
        del_j = np.matmul(np.transpose(trainingSet), del_j)
        del_j = (0.01*(len(trainingSet))*w - del_j)

        new_w = w - 0.5*(1/len(trainingSet))*(del_j)
        weight_dif = np.linalg.norm(new_w-w)
        if(weight_dif<1e-6):
            break
        w = new_w
        #print("Num_epochs, ", num_epochs, "Weight diff ", weight_dif)
        
    fin_res = np.matmul(trainingSet, w)
    sum1= 0
    for i in range(len(w)):
        sum1 += w[i] * trainingSet[0][i]
        
    test_res = np.matmul(testSet, w)
    train_sum = 0
    total = len(fin_res)
    print(w[0])
    print(fin_res)
    for i in range(len(fin_res)):
        if((results[i] == 1 and fin_res[i]>0) or (results[i] == -1 and fin_res[i]<=0)):
            train_sum+=1
    total_test = len(test_res)
    testSum = 0
    for i in range(len(test_res)):
        if((testRes[i] == 1 and test_res[i]>0) or (testRes[i] == -1 and test_res[i]<=0)):
            testSum+=1
    print("Training Accuracy svm: ", train_sum/total)
    print("Testing Accuracy svm: ", testSum/total_test)


if(args.modelIdx == "1"):
    
    lr(train, test)
elif(args.modelIdx == "2"):
    svm(train,test)
