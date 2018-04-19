#based off of the TA's sample code

import numpy as np
import matplotlib.pyplot as plt
import pprint   #used to print 2d arrays nicely

#************************************************************
# QUESTION 0
#
# np.loadtxt() was used to read in data
#
# populateAttributes() was used to read in the attribute names
# reads in file that has the name of each attribute
# used mainly for labeling the axis and graphs
def populateAttributes(fileName):
  file = open(fileName, "r")
  fileBuffer = file.readlines()
  tempList = []
  for line in fileBuffer:
    noNewLine = line.strip('\n')
    if(noNewLine != ""):
      tempList.append(noNewLine)
  return tempList

#************************************************************
# Question 1.1
# Assumption: You want me to make a Histogram for every feature and divde
#             it between classes. Plot each feature for each class of each data set
# topic - "Iris" or "Wine"
# start - index of first item of a class
# end   - index of last item in the same class
def makeHistogram(topic, start, end):
    bins = raw_input("How many bins? ")
    Y = np.zeros(int(bins))

    # iterates through each column
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

        # add a title + axis for which data set
        if(topic == "Iris"):
            plt.xlabel(attributes[colNum] + " in cm")
            plt.ylabel("Frequency")
        elif(topic == "Wine"):
            plt.xlabel(attributes[colNum])
            plt.ylabel("Frequency")

        plt.title(topic)
        plt.show()

# question 1.2
# Assumptions: Pretty much the same as the histogram but make a box plot
def makeBoxPlot(topic, start, end):
    # create a box plot for each feature in each data set
    for colNum in range(dataSet.shape[1]):
        plt.boxplot(dataSet[start:end, colNum], vert=False)
        if(topic == "Iris"):
            plt.xlabel(attributes[colNum] + " in cm")
        elif(topic == "Wine"):
            plt.xlabel(attributes[colNum])

        plt.show()

#************************************************************
# Question 2.1.a
# x - attribute 1, y - attribute 2
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
    # calculate covariance
    for i in range(dataSet.shape[0]):
        val_x = col_x[i] - mean_x
        val_y = col_y[i] - mean_y
        sum_cov = sum_cov + (val_x * val_y)
        sum_std_x = sum_std_x + ((val_x) ** 2)
        sum_std_y = sum_std_y + ((val_y) ** 2)

    cov = sum_cov / dataSet.shape[0]
    # calculate variance
    var_x = sum_std_x / dataSet.shape[0]
    var_y = sum_std_y / dataSet.shape[0]

    # calculate standard deviation
    std_x = (var_x ** (.5))
    std_y = (var_y ** (.5))

    # print cov / (std_x * std_y)
    return cov / (std_x * std_y)

# Question 2.1.b + Question 2.1.d
def makeCorrelationMatrix(isHeatMap):
    correlationMatrix = []
    #initialize a 2d array
    for i in range(dataSet.shape[1]):
        new = []
        for j in range(dataSet.shape[1]):
            new.append(0)
        correlationMatrix.append(new)

    #fill in each cell with correlation
    for rowNum in range(dataSet.shape[1]):
        for colNum in range(dataSet.shape[1]):
            if (rowNum == colNum):
                correlationMatrix[rowNum][colNum] = 1
            else:
                correlationMatrix[rowNum][colNum] = correlation(rowNum, colNum)

    # either make correlation matrix or a heatmap
    # both use the same data
    if(not isHeatMap):
        pprint.pprint(correlationMatrix)
    elif(isHeatMap):
        fig, ax = plt.subplots()
        plt.imshow(np.array(correlationMatrix), cmap='gist_heat', interpolation='nearest')
        plt.xticks(np.arange(dataSet.shape[1]), attributes)
        plt.yticks(np.arange(dataSet.shape[1]), attributes)
        ax.xaxis.tick_top()
        # rotate the ticks so they're not squished together
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        plt.colorbar()
        plt.show()

# Question 2.2.a
# we want to plot
#   1. feature isn't equal to itself
#   2. feature isn't just flipped
#       i.g. sepal length vs sepal width, sepal width vs sepal length
def makeScatterPlot():
    for feature1 in range(dataSet.shape[1]):
        for feature2 in range(dataSet.shape[1]):
            if((feature1 != feature2) and (feature1 < feature2)):
                # features from class 1, 2, 3
                x_1 = dataSet[0:50, feature1]
                y_1 = dataSet[0:50, feature2]
                x_2 = dataSet[50:100, feature1]
                y_2 = dataSet[50:100, feature2]
                x_3 = dataSet[100:150, feature1]
                y_3 = dataSet[100:150, feature2]

                # class 1 = red; class 2 = blue; class 3 = yellow
                colors_1 = ['red']
                colors_2 = ['blue']
                colors_3 = ['yellow']

                # plot everything
                plt.title(attributes[feature1] + " vs " + attributes[feature2] +
                          "\nRed - Class 1; Blue - Class 2; Yellow - Class 3")
                plt.scatter(x_1, y_1, c=colors_1, s=100)
                plt.scatter(x_2, y_2, c=colors_2, s=100)
                plt.scatter(x_3, y_3, c=colors_3, s=100)
                plt.show()

# Question 2.3.a
def distance(x,y,p):
    sum = 0
    for colNum in range(0,dataSet.shape[1]):
        temp = abs(x[colNum] - y[colNum]) ** p
        sum = sum + temp
    dist = sum ** (1.0/p)
    return dist

# Question 2.3.b
def makeDistanceMatrix():
    distMatrix_p1 = []
    distMatrix_p2 = []
    #initialize a 2d array
    for i in range(dataSet.shape[0]):
        new = []
        for j in range(dataSet.shape[0]):
            new.append(0)
        distMatrix_p1.append(new)
        distMatrix_p2.append(new)

    #fill in each cell with distance
    for rowNum in range(dataSet.shape[0]):
        for colNum in range(dataSet.shape[0]):
            if (rowNum == colNum):
                distMatrix_p1[rowNum][colNum] = 0
                distMatrix_p2[rowNum][colNum] = 0
            else:
                distMatrix_p1[rowNum][colNum] = distance(dataSet[rowNum, 0:dataSet.shape[1]], dataSet[colNum, 0:dataSet.shape[1]], 1)
                distMatrix_p2[rowNum][colNum] = distance(dataSet[rowNum, 0:dataSet.shape[1]], dataSet[colNum, 0:dataSet.shape[1]], 2)

    # plot it into a heatmap... for the iris it's gonna be 150x150 matrix
    plt.imshow(np.array(distMatrix_p1), cmap='gist_heat', interpolation='nearest')
    plt.colorbar()
    plt.title("Distance Matrix P = 1")
    plt.xticks(np.arange(25, 150, step=50), ["Class 1", "Class 2", "Class 3"])
    plt.yticks(np.arange(25, 150, step=50), ["Class 1", "Class 2", "Class 3"])
    plt.show()

    plt.imshow(np.array(distMatrix_p2), cmap='gist_heat', interpolation='nearest')
    plt.colorbar()
    plt.title("Distance Matrix P = 2")
    plt.xticks(np.arange(25, 150, step=50), ["Class 1", "Class 2", "Class 3"])
    plt.yticks(np.arange(25, 150, step=50), ["Class 1", "Class 2", "Class 3"])
    plt.show()

# Question 2.3.e
def findNearestNeighbor(topic):
    index_i = -1;
    index_j = -1;
    temp = []
    for i in range(dataSet.shape[0]):
        shortest_distance = float('inf')
        for j in range(dataSet.shape[0]):
            temp_dst = distance(dataSet[i, 0:dataSet.shape[1]], dataSet[j, 0:dataSet.shape[1]], 1)
            if(shortest_distance > temp_dst and j != i):
                shortest_distance = temp_dst
                index_i = i
                index_j = j

        if(topic == "Iris"):
            if(index_j < 50):
                if(index_i < 50):
                    print "Nearest Neighbor Pairs: (" + str(index_i + 1) + ", " + str(index_j + 1) + "). Class of neighbor: Iris-setosa. Same class? Yes"
                else:
                    print "Nearest Neighbor Pairs: (" + str(index_i + 1) + ", " + str(index_j + 1) + "). Class of neighbor: Iris-setosa. Same class? No"
            elif(index_j < 99):
                if(index_i < 99 and index_i > 49):
                    print "Nearest Neighbor Pairs: (" + str(index_i + 1) + ", " + str(index_j + 1) + "). Class of neighbor: Iris-versicolor. Same class? Yes"
                else:
                    print "Nearest Neighbor Pairs: (" + str(index_i + 1) + ", " + str(index_j + 1) + "). Class of neighbor: Iris-versicolor. Same class? No"
            elif(index_j > 98):
                if(index_i > 98):
                    print "Nearest Neighbor Pairs: (" + str(index_i + 1) + ", " + str(index_j + 1) + "). Class of neighbor: Iris-virginica. Same class? Yes"
                else:
                    print "Nearest Neighbor Pairs: (" + str(index_i + 1) + ", " + str(index_j + 1) + "). Class of neighbor: Iris-virginica. Same class? No"
        elif(topic == "Wine"):
            if(index_j < 59):
                if(index_i < 59):
                    print "Nearest Neighbor Pairs: (" + str(index_i + 1) + ", " + str(index_j + 1) + "). Class of neighbor: Class 1. Same class? Yes"
                else:
                    print "Nearest Neighbor Pairs: (" + str(index_i + 1) + ", " + str(index_j + 1) + "). Class of neighbor: Class 1. Same class? No"
            elif(index_j < 129):
                if(index_i < 129 and index_i > 58):
                    print "Nearest Neighbor Pairs: (" + str(index_i + 1) + ", " + str(index_j + 1) + "). Class of neighbor: Class 2. Same class? Yes"
                else:
                    print "Nearest Neighbor Pairs: (" + str(index_i + 1) + ", " + str(index_j + 1) + "). Class of neighbor: Class 2. Same class? No"
            elif(index_j > 128):
                if(index_i > 128):
                    print "Nearest Neighbor Pairs: (" + str(index_i + 1) + ", " + str(index_j + 1) + "). Class of neighbor: Class 3. Same class? Yes"
                else:
                    print "Nearest Neighbor Pairs: (" + str(index_i + 1) + ", " + str(index_j + 1) + "). Class of neighbor: Class 3. Same class? No"



# loads in data for data set
# asks you what you want
userInput = raw_input("Which data set?\n1. Iris\n2. Wine\n")
if(userInput == "1"):
    dataSet = np.loadtxt('iris.data.txt', delimiter=',', usecols=(0, 1, 2, 3))
    attributes = populateAttributes('iris.name.txt')
    whatDo = raw_input("""What do?\n1. Histogram\n2. Box Plot\n3. Correlation Matrix\n4. Scatter Plot\n5. Distance\n""")
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
        heatMap = raw_input("Heatmap?\n1. No, I just want the correlation matrix\n2. Yes, I want a heat map\n")
        if(heatMap == "1"):
            makeCorrelationMatrix(False);
        elif(heatMap == "2"):
            makeCorrelationMatrix(True);
    elif(whatDo == "4"):
        makeScatterPlot()
    elif(whatDo == "5"):
        whatDoMore = raw_input("Distance Matrix?\n1. Yes, I want distance matrix/heat map.\n2. No, I want nearest neighbors\n")
        if(whatDoMore == "1"):
            makeDistanceMatrix()
        elif(whatDoMore == "2"):
            findNearestNeighbor("Iris")

elif(userInput == "2"):
    dataSet = np.loadtxt('wine.data.txt', delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13))
    attributes = populateAttributes('wine.name.txt')
    whatDo = raw_input("What Graph?\n1. Histogram\n2. Box Plot\n3. Correlation Matrix\n4. Distance Matrix\n")
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
        heatMap = raw_input("Heatmap?\n1. No, I just want the correlation matrix\n2. Yes, I want a heat map\n")
        if(heatMap == "1"):
            makeCorrelationMatrix(False);
        elif(heatMap == "2"):
            makeCorrelationMatrix(True);
    elif(whatDo == "4"):
        whatDoMore = raw_input("Distance Matrix?\n1. Yes, I want distance matrix/heat map.\n2. No, I want nearest neighbors\n")
        if(whatDoMore == "1"):
            makeDistanceMatrix()
        elif(whatDoMore == "2"):
            findNearestNeighbor("Wine")
