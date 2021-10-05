import pandas as pd
import numpy as np
import argparse
pd.options.mode.chained_assignment = None
parser = argparse.ArgumentParser(description='')
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

df = pd.read_csv(args.input)
attributes = list(df.columns)
attributes.remove('gender')
attributes.remove('race')
attributes.remove('race_o')
attributes.remove('samerace')
attributes.remove('field')
attributes.remove('decision')
continuous_vals  = attributes

def discrete(df, no_bins):

    bins = {}
    for i in range(len(continuous_vals)):
        bins[continuous_vals[i]] = []
        if('age' in continuous_vals[i]):
            interval = (58-18)/no_bins
            switch_points = []
            start = 18
            while(start<=58):
                switch_points.append(start)
                start = round(start + interval,2)
            
            bins[continuous_vals[i]] = switch_points
        elif('pref_o' in continuous_vals[i] or '_important' in continuous_vals[i]):
            interval = (1)/no_bins
            switch_points = []
            start = 0
            while(start<=1):
                switch_points.append(start)
                start = round(start + interval,2)
            
            bins[continuous_vals[i]] = switch_points
        elif('correlate' in continuous_vals[i]):
            interval = (2)/no_bins
            switch_points = []
            start = -1
            while(start<=1):
                switch_points.append(start)
                start = round(start + interval,2)
            bins[continuous_vals[i]] = switch_points
        else:
            interval = (10)/no_bins
            switch_points = []
            start = 0
            while(start<=10):
                switch_points.append(start)
                start = round(start + interval,2)
            bins[continuous_vals[i]] = switch_points
        bins[continuous_vals[i]][0] -= 10
        bins[continuous_vals[i]][no_bins] **=2
    #print(bins['tvsports'])
    #print(bins)
    #print(df[continuous_vals[0]])
    #print(bins[continuous_vals[0]])
    #temp = pd.cut(df[continuous_vals[0]], bins[continuous_vals[0]], labels=range(5))
    
    for i in continuous_vals:
        bin_range = bins[i]
        df[i] = pd.cut(df[i], bin_range, labels=range(no_bins))
        bin_count = []
        bin_count.append(list(df[i]).count(0))
        bin_count.append(list(df[i]).count(1))
        bin_count.append(list(df[i]).count(2))
        bin_count.append(list(df[i]).count(3))
        bin_count.append(list(df[i]).count(4))
        print(str(i)+':', bin_count)
    """
    for i in continuous_vals:
        bin_range = bins[i]
        for j in range(len(df[i])):
            if(df[i][j] < bin_range[0] or df[i][j] <=bin_range[1]):
                df[i][j] = 0
            else:
                for k in range(2, len(bin_range)):
                    if(df[i][j] <= bin_range[k]):
                        df[i][j] = k-1
                        break
            if(df[i][j] > bin_range[len(bin_range)-1]):
                df[i][j] = len(bin_range)-2
    
        bin_count = []
        bin_count.append(list(df[i]).count(0))
        bin_count.append(list(df[i]).count(1))
        bin_count.append(list(df[i]).count(2))
        bin_count.append(list(df[i]).count(3))
        bin_count.append(list(df[i]).count(4))
        print(str(i)+':', bin_count)
    """
    return df
df = discrete(df, 5)
df.to_csv(args.output, index=False)

        
