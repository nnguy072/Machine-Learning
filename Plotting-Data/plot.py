#based off of the TA's sample code

import numpy as np
import matplotlib.pyplot as plt
import pprint

def populateAttributes(fileName):
  file = open(fileName, "r")
  fileBuffer = file.readlines()
  tempList = []
  for line in fileBuffer:
    noNewLine = line.strip('\n')
    if(noNewLine != ""):
      tempList.append(noNewLine)
  return tempList

def makeHistogram(topic, start, end):
    bins = raw_input("How many bins? ")
    Y = np.zeros(int(bins))

    for colNum in range(dataSet.shape[1]):
        col = dataSet[start:end, colNum]
        min_col = min(col)
        max_col = max(col)
        binRange = float((max_col - min_col) / int(bins))
        print binRange

        leftRange = min_col
        rightRange = min_col + binRange

        X = []
        for i in range(int(bins)):
            freq = col[col > leftRange]
            freq = freq[freq <= rightRange]
            Y[i] = freq.size

            x_range = str(leftRange) + "-" + str(rightRange)
            X.append(x_range)
            leftRange = rightRange
            rightRange = rightRange + binRange

        inds = np.arange(int(bins))
        width = .5
        plotCol = plt.bar(inds, Y, width)
        plt.xticks(inds, X)

        if(topic == "Iris"):
            plt.xlabel(attributes[colNum] + " in cm")
            plt.ylabel("Frequency")
        elif(topic == "Wine"):
            plt.xlabel(attributes[colNum])
            plt.ylabel("Frequency")

        plt.title(topic)
        plt.show()

def makeBoxPlot(topic, start, end):
    for colNum in range(dataSet.shape[1]):
        plt.boxplot(dataSet[start:end, colNum], vert=False)
        if(topic == "Iris"):
            plt.xlabel(attributes[colNum] + " in cm")
        elif(topic == "Wine"):
            plt.xlabel(attributes[colNum])

        plt.show()

# x - attrivute 1, y - attribute 2
def correlation(x, y):
    col_x = dataSet[0:dataSet.shape[0], x]
    col_y = dataSet[0:dataSet.shape[0], y]

    # calculate mean for both attributes
    sum_x = 0
    sum_y = 0
    for i in range(dataSet.shape[0]):
        sum_x = sum_x + col_x[i]
        sum_y = sum_y + col_y[i]

    mean_x = sum_x / col_x.shape[0]
    mean_y = sum_y / col_y.shape[0]

    sum_cov = 0
    sum_std_x = 0
    sum_std_y = 0
    for i in range(dataSet.shape[0]):
        val_x = col_x[i] - mean_x
        val_y = col_y[i] - mean_y
        sum_cov = sum_cov + (val_x * val_y)
        sum_std_x = sum_std_x + ((val_x) ** 2)
        sum_std_y = sum_std_y + ((val_y) ** 2)

    cov = sum_cov / dataSet.shape[0]
    var_x = sum_std_x / dataSet.shape[0]
    var_y = sum_std_y / dataSet.shape[0]

    std_x = (var_x ** (.5))
    std_y = (var_y ** (.5))

    # print cov / (std_x * std_y)
    return cov / (std_x * std_y)

def makeCorrelationMatrix():
    correlationMatrix = []
    for i in range(dataSet.shape[1]):
        new = []
        for j in range(dataSet.shape[1]):
            new.append(0)
        correlationMatrix.append(new)

    for rowNum in range(dataSet.shape[1]):
        for colNum in range(dataSet.shape[1]):
            if (rowNum == colNum):
                correlationMatrix[rowNum][colNum] = 1
            else:
                correlationMatrix[rowNum][colNum] = correlation(rowNum, colNum)

    pprint.pprint(correlationMatrix)

userInput = raw_input("Which data set?\n1. Iris\n2. Wine\n")
if(userInput == "1"):
    dataSet = np.loadtxt('iris.data.txt', delimiter=',', usecols=(0, 1, 2, 3))
    attributes = populateAttributes('iris.name.txt')
    whatDo = raw_input("What do?\n1. Histogram\n2. Box Plot\n3. Correlation Matrix\n")
    if(whatDo == "1"):
        whatClass = raw_input("What Class?\n1. Iris-setosa\n2. Iris-versicolor\n3. Iris-virginica\n")
        if(whatClass == "1"):
            makeHistogram("Iris", 0, 50)
        elif(whatClass == "2"):
            makeHistogram("Iris", 50, 100)
        elif(whatClass == "3"):
            makeHistogram("Iris", 100, 150)
    elif(whatDo == "2"):
        whatClass = raw_input("What Class?\n1. Iris-setosa\n2. Iris-versicolor\n3. Iris-virginica\n")
        if(whatClass == "1"):
            makeBoxPlot("Iris", 0, 50)
        elif(whatClass == "2"):
            makeBoxPlot("Iris", 50, 100)
        elif(whatClass == "3"):
            makeBoxPlot("Iris", 100, 150)
    elif(whatDo == "3"):
        makeCorrelationMatrix();
elif(userInput == "2"):
    dataSet = np.loadtxt('wine.data.txt', delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
    attributes = populateAttributes('wine.name.txt')
    whatDo = raw_input("What Graph?\n1. Histogram\n2. Box Plot\n3. Correlation Matrix\n")
    if(whatDo == "1"):
        whatClass = raw_input("What Class?\n1. Class 1\n2. Class 2\n3. Class 3\n")
        if(whatClass == "1"):
            makeHistogram("Wine", 0, 59)
        elif(whatClass == "2"):
            makeHistogram("Wine", 59, 130)
        elif(whatClass == "3"):
            makeHistogram("Wine", 130, 178)
    elif(whatDo == "2"):
        whatClass = raw_input("What Class?\n1. Class 1\n2. Class 2\n3. Class 3\n")
        if(whatClass == "1"):
            makeBoxPlot("Wine", 0, 59)
        elif(whatClass == "2"):
            makeBoxPlot("Wine", 59, 130)
        elif(whatClass == "3"):
            makeBoxPlot("Wine", 130, 178)
    elif(whatDo == "3"):
        makeCorrelationMatrix();

