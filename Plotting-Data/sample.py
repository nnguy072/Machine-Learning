import numpy as np
import matplotlib.pyplot as plt

# load everything besides the word at the end
a = np.loadtxt('iris.data.txt', delimiter=',', usecols=(0, 1, 2, 3))
b = a[0:50, 0]  # get 50 of the first column
print b.size
min_b = min(b)  # get min of 50 in the first column
max_b = max(b)  #get max of 50 in the first column
bins = 5
binRange = float((max_b - min_b) / bins)    #how big each thing is width wise i.g. 3,4,5
Y = np.zeros(bins)  #holds how many items are inside range
#{0 , 0 , 0, 0 , 0}

#make the interval
r1 = min_b
r2 = min_b + binRange

X = []
for i in range(bins):
    c = b[b > r1]   #anything bigger than min(b)
    c = c[c <= r2]  #anything less than or equal to min_b + binRange
    Y[i] = c.size   #count how big c is basically
    string = str(r1) + "-" + str(r2)    #min - max
    X.append(string)
    r1 = r2
    r2 = r2 + binRange

#{0 , 1 , 2 , 3, 4}
inds = np.arange(bins)
width = 0.50
p1 = plt.bar(inds, Y, width)
print X
plt.xticks(inds, X)
plt.show()