import pandas as pd
import copy
import numpy as np
import math
from statistics import mean, stdev
import matplotlib.pyplot as plt
import json

train_nbc = pd.read_csv('trainingSet_NBC.csv')
train = pd.read_csv('trainingSet.csv')

train_nbc = train_nbc.sample(random_state=18, frac = 1)
train = train.sample(random_state = 18, frac=1)

train_slices = []
train_nbc_slices = []
for i in range(0,len(train),520):
    train_slices.append(train[i:i+520])
    train_nbc_slices.append(train_nbc[i:i+520])
t_frac = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
non_cont = ['gender', 'race', 'race_o', 'samerace', 'field', 'decision']
def nbc(training, testing):

    training.reset_index(inplace=True, drop=True)
    testing.reset_index(inplace=True, drop=True)
    positive_class = training.loc[training['decision'] == 1]
    negative_class = training.loc[training['decision'] == 0]

    prior = {}
    prior[1] = len(positive_class)/len(training)
    prior[0] = len(negative_class)/len(training)
    
    
    cond_prob_matrix_pos = {}
    cond_prob_matrix_neg = {}
    len_pos = len(positive_class)
    len_neg = len(negative_class)
    attribs = list(training.columns)
    laplace = {}
    for i in attribs:
        if(i == 'decision'):
            break
        attrib_values = list(set(list(training[i])))
        if i in non_cont:
            laplace[i] = len(attrib_values)
        else:
            laplace[i] = 5
        cond_prob_matrix_pos[i] = {}
        cond_prob_matrix_neg[i] = {}
        for j in attrib_values:
            cond_prob_matrix_pos[i][j] = (list(positive_class[i]).count(j)+1)/(laplace[i] + len_pos)
            cond_prob_matrix_neg[i][j] = (list(negative_class[i]).count(j)+1)/(laplace[i] + len_neg)

    count = 0
    for i in range(len(training)):
        init_prob_pos = prior[1]
        init_prob_neg = prior[0]
        for j in attribs:
            if(j == 'decision'):
                break
            init_prob_pos = init_prob_pos * cond_prob_matrix_pos[j][training[j][i]]
            init_prob_neg = init_prob_neg * cond_prob_matrix_neg[j][training[j][i]]
        frac_pos = init_prob_pos/(init_prob_pos + init_prob_neg)
        frac_neg = init_prob_neg/(init_prob_pos + init_prob_neg)
        if(frac_pos > frac_neg):
            decision = 1
        else:
            decision = 0
        
        if(decision == training['decision'][i]):
            count+=1
        
    #print("Training Accuracy: %0.2f"%(count/len(training)))
    training_acc = count/len(training)
    count = 0
    for i in range(len(testing)):
        init_prob_pos = prior[1]
        init_prob_neg = prior[0]
        for j in attribs:
            if(j == 'decision'):
                break
            try:
                init_prob_pos = init_prob_pos * cond_prob_matrix_pos[j][testing[j][i]]
            except:
                init_prob_pos = init_prob_pos * (1/laplace[j])
            try:
                init_prob_neg = init_prob_neg * cond_prob_matrix_neg[j][testing[j][i]]
            except:
                init_prob_neg = init_prob_neg * (1/laplace[j])
        frac_pos = init_prob_pos/(init_prob_pos + init_prob_neg)
        frac_neg = init_prob_neg/(init_prob_pos + init_prob_neg)
        if(frac_pos > frac_neg):
            decision = 1
        else:
            decision = 0
        
        if(decision == testing['decision'][i]):
            count+=1

    #print("Testing Accuracy: %0.2f"%(count/len(testing)))
    test_acc = count/len(testing)
    return training_acc, test_acc

def lr(trainingSet, testSet):
    trainingSet.reset_index(inplace=True, drop=True)
    testSet.reset_index(inplace=True, drop=True)

    trainingSet = trainingSet.to_numpy()
    testSet = testSet.to_numpy()
    length = len(trainingSet[0]) -1
    w = np.array([0 for i in range(length)])
    num_epochs = 0
    tol = 1e-6
    weight_dif = 1
    trainMat = np.matrix(trainingSet)
    results = trainMat[:, [len(trainingSet[0])-1]]
    results = np.squeeze(np.asarray(results))
    trainingSet = np.delete(trainingSet, len(trainingSet[0])-1, 1)

    
    testRes = np.matrix(testSet)[:,[len(testSet[0])-1]]
    testRes = np.squeeze(np.asarray(testRes))
    testSet = np.delete(testSet, len(testSet[0])-1, 1)

    for epochs in range(500):
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
        if(weight_dif<tol):
            break
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
    #print("Training Accuracy LR: ", train_sum/total)
    #print("Testing Accuracy LR: ", testSum/total_test)
    return train_sum/total, testSum/total_test
    
def svm(trainingSet, testSet):
    trainingSet.reset_index(inplace=True, drop=True)
    testSet.reset_index(inplace=True, drop=True)
    train_cols = list(trainingSet.columns)
    trainingSet = trainingSet.to_numpy()
    testSet = testSet.to_numpy()
    #print(trainingSet)
    #print(trainingSet[0])
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
    for i in range(500):
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
        
    fin_res = np.matmul(trainingSet, w)
    sum1= 0
    for i in range(len(w)):
        sum1 += w[i] * trainingSet[0][i]
    test_res = np.matmul(testSet, w)
    train_sum = 0
    total = len(fin_res)
    for i in range(len(fin_res)):
        if((results[i] == 1 and fin_res[i]>0) or (results[i] == -1 and fin_res[i]<=0)):
            train_sum+=1
    total_test = len(test_res)
    testSum = 0
    for i in range(len(test_res)):
        if((testRes[i] == 1 and test_res[i]>0) or (testRes[i] == -1 and test_res[i]<=0)):
            testSum+=1
    return train_sum/total, testSum/total_test



results = {}

for i in t_frac:
    accuracy_svm = []
    accuracy_nbc = []
    accuracy_lr = []
    for j in range(10):
        test_set_nbc = train_nbc_slices[j]
        test_set = train_slices[j]
        new_train = copy.copy(train_slices)
        
        new_train.pop(j)
        new_train_nbc = copy.copy(train_nbc_slices)
        new_train_nbc.pop(j)
        
        new_train = pd.concat(new_train)
        new_train_nbc = pd.concat(new_train_nbc)
        new_train = new_train.sample(random_state=32, frac = i)
        new_train_nbc = new_train_nbc.sample(random_state=32, frac=i)
        accuracy_svm.append(svm(new_train, test_set)[1])
        accuracy_nbc.append(nbc(new_train_nbc, test_set_nbc)[1])
        accuracy_lr.append(lr(new_train, test_set)[1])
    nbc_avg_acc = mean(accuracy_nbc)
    nbc_std = stdev(accuracy_nbc)
    svm_avg_acc = mean(accuracy_svm)
    svm_std = stdev(accuracy_svm)
    lr_avg_acc = mean(accuracy_lr)
    lr_std = stdev(accuracy_lr)
    nbc_std_err = nbc_std/math.sqrt(10)
    svm_std_err = svm_std/math.sqrt(10)
    lr_std_err = lr_std/math.sqrt(10)
    results[i] = {}
    results[i]['svm'] = [svm_std_err, svm_avg_acc]
    results[i]['lr'] = [lr_std_err, lr_avg_acc]
    results[i]['nbc'] = [nbc_std_err, nbc_avg_acc]
    print("NBC Avg acc ", nbc_avg_acc, " NBC std err ", nbc_std_err)
    print("LR Avg acc ", lr_avg_acc, " LR std err ", lr_std_err)
    print("SVM Avg acc ", svm_avg_acc, " SVM std err ", svm_std_err)

fracs = [4680 * i for i in list(results.keys())]
svm_vals = [results[i]['svm'][1] for  i in results.keys()]
nbc_vals = [results[i]['nbc'][1] for  i in results.keys()]
lr_vals = [results[i]['lr'][1] for  i in results.keys()]

svm_errs = [results[i]['svm'][0] for  i in results.keys()]
nbc_errs = [results[i]['nbc'][0] for  i in results.keys()]
lr_errs = [results[i]['lr'][0] for  i in results.keys()]

svm_error = [np.subtract(svm_vals,svm_errs), np.add(svm_vals,svm_errs)]
nbc_error = [np.subtract(nbc_vals,nbc_errs), np.add(nbc_vals,nbc_errs)]
lr_error = [np.subtract(lr_vals,lr_errs), np.add(lr_vals,lr_errs)]

with open('acc_err.json', 'w') as fp:
    json.dump(results, fp)
print(svm_errs)
plt.errorbar(fracs, svm_vals, fmt='blue', yerr=svm_errs, label='svm')
plt.errorbar(fracs, nbc_vals, fmt='red', yerr=nbc_errs,label='nbc')
plt.errorbar(fracs, lr_vals, fmt='green', yerr=lr_errs, label='lr')
plt.xlabel('train size')
plt.ylabel('accuracy')
plt.legend()
plt.show()

    
    
    
