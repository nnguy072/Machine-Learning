import numpy as np
import matplotlib.pyplot as plt

dataSet = np.loadtxt('iris.data.txt', delimiter=',', usecols=(0, 1, 2, 3))

plt.hist(dataSet)
plt.show()