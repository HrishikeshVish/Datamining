import json
from scipy.stats import ttest_1samp, ttest_ind
from statistics import mean
with open('acc_err.json', 'r') as json_file:
    res_data = json.load(json_file)
fracs = res_data.keys()
svm_accs = []
svm_std = []
nbc_accs = []
nbc_std = []
for i in fracs:
    nbc_accs.append(res_data[i]['nbc'][1])
    svm_accs.append(res_data[i]['svm'][1])
    nbc_std.append(res_data[i]['nbc'][0]*(10**0.5))
    svm_std.append(res_data[i]['svm'][0]*(10**0.5))

nbc_avg = mean(nbc_accs)
svm_avg = mean(svm_accs)

nbc_dev = mean(nbc_std)
svm_dev = mean(svm_std)

tset, pval = ttest_1samp(nbc_accs, svm_avg)
tset, pval = ttest_ind(nbc_accs, svm_accs)
print(pval)
if(pval<0.05):
    print("NBC performs better than SVM")
else:
    print("We cannot conclude that NBC performs better than SVM")

    
