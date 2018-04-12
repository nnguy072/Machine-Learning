import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('iris.data.txt', delimiter=',', usecols=(0, 1, 2, 3))
b = a[0:50, 0]
print b.size
min_b = min(b)
max_b = max(b)
bins = 5
binRange = float((max_b - min_b) / bins)
Y = np.zeros(bins)

r1 = min_b
r2 = min_b + binRange
X = []
for i in range(bins):
    c = b[b > r1]
    c = c[c <= r2]
    Y[i] = c.size
    string = str(r1) + "-" + str(r2)
    X.append(string)
    r1 = r2
    r2 = r2 + binRange


inds = np.arange(bins)
width = 0.35
p1 = plt.bar(inds, Y, width)
print X
plt.xticks(inds, X)
plt.show()