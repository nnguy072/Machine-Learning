import numpy as np
import matplotlib.pyplot as plt
import pprint   #used to print 2d arrays nicely
np.set_printoptions(threshold=np.inf)   #need this to print out all the data for testing

# *********************************************************************
# QUESTION 0: Getting Real Data

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
    return temp1
# *********************************************************************



# *********************************************************************
# QUESTION 1: k-nearest neighborso classifier

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
    print "Doing knn classifier..."
    for i in range(0, len(x_test)):
        dist_dict = {}
        # population dictionary w/ distances
        for j in range(0, len(x_train)):
            dist_dict[j] = distance(x_test[i], x_train[j], p)
        # I found this sort function on stack overflow
        # it sorts the hashtable's value in ascending order
        # i added the [:k] to only get k nearest neighbors
        K_nearestNeighbors = sorted(dist_dict, key=dist_dict.get)[:k]
        y_pred.append(findMajority(K_nearestNeighbors, y_train))
    return np.array(y_pred)
# *********************************************************************



# *********************************************************************
# QUESTION 2

# returns tuple (Accuracy, Sensitivity, Specificity)
def calculateMetrics(y_actual, y_pred):
    total_correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for x in range(0, y_pred.shape[0]):
        if(y_pred[x] == y_actual[x]):
            total_correct = total_correct + 1
        if(y_pred[x] == y_actual[x] and y_pred[x] == 4):
            true_positive = true_positive + 1
        if(y_pred[x] == y_actual[x] and y_pred[x] == 2):
            true_negative = true_negative + 1
        if(y_pred[x] != y_actual[x] and y_pred[x] == 4 and y_actual[x] == 2):
            false_positive = false_positive + 1
        if(y_pred[x] != y_actual[x] and y_pred[x] == 2 and y_actual[x] == 4):
            false_negative = false_negative + 1

    accuracy = float(total_correct) / y_pred.shape[0]
    sensitivity = float(true_positive) / (true_positive + false_negative)
    specificity = float(true_negative) / (true_negative + false_positive)

    print "Accuracy: " + str(total_correct) + "/" + str(y_pred.shape[0]) + " = " + str(accuracy)
    print "Sensitivity: " + str(true_positive) + "/(" + str(true_positive) +  " + " + str(false_negative) + ") = " + str(sensitivity)
    print "Specificity: " + str(true_negative) + "/(" + str(true_negative) + " + " + str(false_positive) + ") = " + str(specificity)
    print ""
    return (accuracy, sensitivity, specificity)


# kfcv = k-fold-cross-validation; we'll be using 10 fold
# basically, you want to split data into k equal sections
# use one section as test and rest as training. then repeat
# until you use all of them.
# TODO: make more robost by passing in function pointer for what classifer
def kfcv(dataSet, k):
    # TODO: evenly divide sections. if number is not evenly divisble, we'll
    #       miss some data points
    k_fold_sections = dataSet.shape[0] / k
    left = 0
    acc_list = []
    sens_list = []
    spec_list = []
    num_folds = []
    count = 1

    # basically we move sections by doing something like
    # (0 to n) -> (n to n * 2) -> (n * 2) to (n * 3); etc etc
    # i.g. if we had 680, it would be 0-68, 68-136, 136-204
    # each section is equal 68
    # kind of like a sliding window algorithm
    lp_p_val = input("Enter Lp P Value: ")
    k_val = input("Enter K value: ")
    print "Doing Cross Fold Validation..."
    for x in range(1, k + 1):
        testSet = dataSet[left:k_fold_sections * x, :(dataSet.shape[1] - 1)]
        # removes the section that is suppose to be the test set from the trianing data
        # then remove class labels from training set and make a new vector that holds that class data
        trainingSet = np.delete(dataSet, slice(left, (k_fold_sections * x)), axis=0)[:, :(dataSet.shape[1] - 1)]
        trainingSetLabls = np.delete(dataSet, slice(left, (k_fold_sections * x)), axis=0)[:, (dataSet.shape[1] - 1):]
        y_pred = knn_classifier(testSet, trainingSet, trainingSetL  abls, lp_p_val, k_val)
        y_actual = dataSet[left:k_fold_sections * x, (dataSet.shape[1] - 1):]

        print "Predicted Classes"
        print y_pred
        print "Actual Classes"
        print dataSet[left:k_fold_sections * x, (dataSet.shape[1] - 1):].reshape(1, -1)
        metrics = calculateMetrics(y_actual ,y_pred)
        acc_list.append(metrics[0])
        sens_list.append(metrics[1])
        spec_list.append(metrics[2])
        num_folds.append(count)
        count = count + 1

        left = k_fold_sections * x  # move to next section

    # convert to numpy array to do other calculations
    acc_array = np.array(acc_list)
    sens_array = np.array(sens_list)
    spec_array = np.array(spec_list)
    print "Accuracy Mean/Standard Deviation   : " + str(acc_array.mean()) + " | " + str(acc_array.std())
    print "Sensitivity Mean/Standard Deviation: " + str(sens_array.mean()) + " | " + str(sens_array.std())
    print "Specificity Mean/Standard Deviation: " + str(spec_array.mean()) + " | " + str(spec_array.std())

    # plt.plot(acc_array)
    plt.title("Nearest Neighbors x Accuracy; P = " + str(lp_p_val) + " K = " + str(k_val))
    plt.ylabel("Accuracy")
    plt.xlabel("Number of folds: " + str(k))
    plt.errorbar(num_folds, acc_array, yerr=acc_array.std(), fmt='o')
    plt.show()

    # plt.plot(sens_array)
    plt.title("Nearest Neighbors x Sensitivity; P = " + str(lp_p_val) + " K = " + str(k_val))
    plt.ylabel("Sensitivity")
    plt.xlabel("Number of folds: " + str(k))
    plt.errorbar(num_folds, sens_array, yerr=sens_array.std(), fmt='o')
    plt.show()

    # plt.plot(spec_array)
    plt.title("Nearest Neighbors x Specificity; P = " + str(lp_p_val) + " K = " + str(k_val))
    plt.ylabel("Specificity")
    plt.xlabel("Number of folds: " + str(k))
    plt.errorbar(num_folds, spec_array, yerr=spec_array.std(), fmt='o')
    plt.show()



# *********************************************************************

def main():
    # get raw data from file
    dataSet = np.array(populateDataSet('breast-cancer-wisconsin.data'))

    whatDo = raw_input("What do?\n1. K-Nearest Neighbors\n2. K-Fold Cross-Validation\n")
    if(whatDo == "1"):
        # get P and K value
        lp_p_val = input("Enter Lp P value: ")
        k_val = input("Enter K value: ")

        # split data into training and test
        # Prof wants to split 80/20; 80 is training 20 is test
        trainingSet = dataSet[:int(len(dataSet) * .8), :(dataSet.shape[1] - 1)]
        trainingSetLabls = dataSet[:int(len(dataSet) * .8), (dataSet.shape[1] - 1):]
        testSet = dataSet[int(len(dataSet) * .8):, :(dataSet.shape[1] - 1)]

        y_pred = knn_classifier(testSet, trainingSet, trainingSetLabls, k_val, lp_p_val)
        # pprint.pprint(y_pred)
        print y_pred
    elif(whatDo == "2"):
        print "Shuffling Data Set..."
        np.random.shuffle(dataSet)
        print "Shuffled."
        kfcv(dataSet, 10)

if __name__ == "__main__":
    main()