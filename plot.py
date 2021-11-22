import matplotlib.pyplot as plt
import json

with open('results.json', 'r') as fp:
    data = json.load(fp)
dataset1SC = [data['1']['2']['sc'], data['1']['4']['sc'], data['1']['8']['sc'], data['1']['16']['sc'],data['1']['32']['sc']]
dataset1WC = [data['1']['2']['wc'], data['1']['4']['wc'], data['1']['8']['wc'], data['1']['16']['wc'],data['1']['32']['wc']]

dataset2SC = [data['2']['2']['sc'], data['2']['4']['sc'], data['2']['8']['sc'], data['2']['16']['sc'],data['2']['32']['sc']]
dataset2WC = [data['2']['2']['wc'], data['2']['4']['wc'], data['2']['8']['wc'], data['2']['16']['wc'],data['2']['32']['wc']]

dataset3SC = [data['3']['2']['sc'], data['3']['4']['sc'], data['3']['8']['sc'], data['3']['16']['sc'],data['3']['32']['sc']]
dataset3WC = [data['3']['2']['wc'], data['3']['4']['wc'], data['3']['8']['wc'], data['3']['16']['wc'],data['3']['32']['wc']]

plt.plot([2,4,8,16,32], dataset1SC,  color='blue', label='dataset1')
plt.plot([2,4,8,16,32], dataset2SC,  color='green', label='dataset2')
plt.plot([2,4,8,16,32], dataset3SC,  color='red', label='dataset3')
plt.xlabel('K Values')
plt.ylabel("Silhouette Coefficient")
plt.legend()
plt.show()


plt.plot([2,4,8,16,32], dataset1WC,  color='blue', label='dataset1')
plt.plot([2,4,8,16,32], dataset2WC,  color='green', label='dataset2')
plt.plot([2,4,8,16,32], dataset3WC,  color='red', label='dataset3')
plt.xlabel('K Values')
plt.ylabel("WC-SSD")
plt.legend()
plt.show()
