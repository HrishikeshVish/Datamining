import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
trainingSet = pd.read_csv('trainingSet.csv')
testSet = pd.read_csv('testSet.csv')
Y = trainingSet['decision']
X = trainingSet.drop('decision', axis=1)
print(X)
print(Y)
clf = RandomForestClassifier(n_estimators=30, bootstrap=False)
#clf = DecisionTreeClassifier(max_depth=8,  min_samples_split=50)
clf.fit(X, Y)
y = clf.predict(X)
count = 0
for i in range(len(y)):
    if(y[i] == Y[i]):
        count +=1
print("Training Accuracy ", count/len(X))
        
