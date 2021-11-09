import pandas as pd
import numpy as np
import copy

trainingSet = pd.read_csv('trainingSet.csv')
testSet = pd.read_csv('testSet.csv')

def perceptron(trainingSet, testSet):
    train_y = copy.copy(np.squeeze(np.asarray(trainingSet['decision'])))
    train_x = trainingSet.drop('decision', axis=1)
    weights = np.zeros((len(list(train_x.columns))))
    bias = 0
    for i in range(len(train_y)):
        if(train_y[i] == 0):
            train_y[i] = -1
    data_len = len(train_x)
    for i in range(5):
        count = 0
        print("Epoch ", i)
        iter_count = 0
        for index, row in train_x.iterrows():
            pred_y = np.squeeze(np.matmul(row, weights)) + bias
            pos_exp = np.exp(pred_y)
            neg_exp = np.exp(-1*pred_y)
            tanh = (pos_exp - neg_exp)/(pos_exp + neg_exp)
            pred_y = tanh
            
            expected_y = train_y[iter_count]
            iter_count +=1
            loss = pred_y - expected_y
            weights = weights - loss*0.01*row
            bias = bias - loss*0.01
    return weights

def calcAccuracy(data, weights):
    expected = np.squeeze(np.asarray(data['decision']))
    test = data.drop('decision', axis=1)
    predict = np.matmul(test, weights)
    count = 0
    for i in range(len(predict)):
        if(predict[i]<0 and expected[i] == 0):
            count +=1
        if(predict[i]>=0 and expected[i] == 1):
            count+=1
    return count/len(predict)
weights = perceptron(trainingSet, testSet)
print("Training Accuracy %0.2f"%(calcAccuracy(trainingSet, weights)))
print("Test Accuracy %0.2f"%(calcAccuracy(testSet, weights)))    
