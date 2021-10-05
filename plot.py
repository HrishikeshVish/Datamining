import matplotlib.pyplot as plt

bin_sizes = [2, 5, 10, 50, 100, 200]
train_acc = [0.75, 0.78, 0.79, 0.80, 0.79, 0.80]
test_acc = [0.72, 0.75, 0.75, 0.75, 0.75, 0.75]

plt.plot(bin_sizes, train_acc, 'blue', label='train')
plt.plot(bin_sizes, test_acc, 'orange', label='test')
plt.xlabel('bin_sizes')
plt.ylabel('accuracy')
plt.legend()
plt.show()

frac = [0.01, 0.1, 0.2, 0.5, 0.6, 0.75, 0.9, 1]
train_acc = [0.93, 0.84, 0.79, 0.79, 0.78, 0.77, 0.77, 0.78]
test_acc = [0.67, 0.74, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

plt.plot(frac, train_acc, 'blue', label='train')
plt.plot(frac, test_acc, 'orange', label='test')
plt.xlabel('frac')
plt.ylabel('accuracy')
plt.legend()
plt.show()
