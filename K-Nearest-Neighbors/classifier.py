import numpy as np
import matplotlib.pyplot as plt
import pprint   #used to print 2d arrays nicely
np.set_printoptions(threshold=np.inf)   #need this to print out all the data for testing

# *********************************************************************
# Question 0: Getting Real Data

# basically read in everything and
# then replace every "?" w/ [TODO: something]
def handleMissingValues(line):
    for x in range(0, len(line)):
        if(line[x] == "?"):
            line[x] = -1    #something here; regression
    return line

# read in raw data from file
def populateDataSet(fileName):
    file = open(fileName, "r")
    temp = file.readlines()
    temp1 = []
    for line in temp:
        noNewLine = line.strip('\n')
        if("?" not in line):
            intList = map(int, noNewLine.split(",")[1:]) #convert from string to int
            temp1.append(intList)
        # uhh TODO: Later when I figure out this handling missing values thing
        # if("?" in line):
        #     intList = map(int, handleMissingValues(noNewLine.split(","))[1:]) #convert from string to int
        #     temp1.append(handleMissingValues(noNewLine.split(","))[1:])
        # else:
        #     intList = map(int, noNewLine.split(",")[1:]) #convert from string to int
        #     temp1.append(intList)
    return temp1
# *********************************************************************



# *********************************************************************
# Question 1: k-nearest neighbor classifier

# Question 2.3.a in Assignment 1
# LP norm; inputs are:
#   x =
#   y =
#   p = Lp distance (1 = manhattan, 2 = euclidean) 4/6/2018 slide
def distance(x,y,p):
    sum = 0
    for colNum in range(0,len(x)):
        temp = abs(x[colNum] - y[colNum]) ** p
        sum = sum + temp
    dist = sum ** (1.0/p)
    return dist

def findMajority(index_of_nearest, label_list):
    nearest_labels = []
    for indexes in index_of_nearest:
        nearest_labels.append(label_list[indexes]);
    beign_count = nearest_labels.count(2);
    malign_count = nearest_labels.count(4);
    if(beign_count >= malign_count):
        return 2
    elif(beign_count < malign_count):
        return 4

# want to calculate distance
# made a dictionary w/ key:value = index_of_y_train:distance
# sort the distance
# get the top k points (indexes)
# get majority class
# y_pred = list of predicted classes
def knn_classifier(x_test, x_train, y_train, k, p):
    y_pred = []
    for i in range(0, len(x_test)):
        dist_dict = {}
        # population dictionary w/ distances
        for j in range(0, len(x_train)):
            dist_dict[j] = distance(x_test[i], x_train[j], p)
        # I found this sort function on stack overflow
        K_nearestNeighbors = sorted(dist_dict, key=dist_dict.get)[:k]
        y_pred.append(findMajority(K_nearestNeighbors, y_train))
    return y_pred
# *********************************************************************

def main():
    # get raw data from file
    dataSet = np.array(populateDataSet('breast-cancer-wisconsin.data'))

    # get P and K value
    lp_p_val = input("Enter Lp P value: ")
    k_val = input("Enter K value: ")

    # split data into training and test
    trainingSet = dataSet[:(len(dataSet) / 2), :(dataSet.shape[1] - 1)]
    trainingSetLabls = dataSet[:(len(dataSet) / 2), (dataSet.shape[1] - 1):]
    testSet = dataSet[(len(dataSet) / 2):, :(dataSet.shape[1] - 1)]

    y_pred = knn_classifier(testSet, trainingSet, trainingSetLabls, k_val, lp_p_val)
    # pprint.pprint(y_pred)
    print y_pred

if __name__ == "__main__":
    main()