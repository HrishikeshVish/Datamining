import pandas as pd

df = pd.read_csv('dating-binned.csv')

test_sample = df.sample(random_state=47, frac=0.2)
indexes = list(test_sample.index)
train_sample = df.drop(indexes)
print(len(train_sample))
print(len(test_sample))
test_sample.to_csv('testSet.csv')
train_sample.to_csv('trainingSet.csv')
#print(df.loc[529] == test_sample.loc[529])
#for index, row in df.iterrows():
    

