import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

df = pd.read_csv('dating.csv')
attributes = list(df.columns)
attributes.remove('gender')
attributes.remove('race')
attributes.remove('race_o')
attributes.remove('samerace')
attributes.remove('field')
attributes.remove('decision')
continuous_vals  = attributes

#print(continuous_vals)
#print(len(continuous_vals))

bins = {}
for i in range(len(continuous_vals)):
    bins[continuous_vals[i]] = []
    if('age' in continuous_vals[i]):
        bins[continuous_vals[i]] = [18, 26, 34, 42, 50, 58]
    elif('pref_o' in continuous_vals[i] or '_important' in continuous_vals[i]):
        bins[continuous_vals[i]] = [0, 0.2, 0.4, 0.6, 0.8, 1]
    elif('correlate' in continuous_vals[i]):
        bins[continuous_vals[i]] = [-1, -0.6, -0.2, 0.2, 0.6, 1]
    else:
        bins[continuous_vals[i]] = [0, 2, 4, 6, 8, 10]


        
for i in continuous_vals:
    bin_range = bins[i]
    for j in range(len(df[i])):
        if(df[i][j] < bin_range[0] or df[i][j] <bin_range[1]):
            df[i][j] = 0
        elif(df[i][j] <bin_range[2]):
            df[i][j] = 1
        elif(df[i][j] <bin_range[3]):
            df[i][j] = 2
        elif(df[i][j] <bin_range[4]):
            df[i][j] = 3
        elif(df[i][j]<=bin_range[5] or df[i][j]>bin_range[5]):
            df[i][j] = 4
    bin_count = []
    bin_count.append(list(df[i]).count(0))
    bin_count.append(list(df[i]).count(1))
    bin_count.append(list(df[i]).count(2))
    bin_count.append(list(df[i]).count(3))
    bin_count.append(list(df[i]).count(4))
    print(str(i)+':', bin_count)
df.to_csv('dating-binned.csv', index=False)

        
