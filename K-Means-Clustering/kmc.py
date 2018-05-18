import numpy as np
import matplotlib.pyplot as plt
import pprint   #used to print 2d arrays nicely

# ripped from assignment 1 and 2
def distance(x,y,p):
    sum = 0
    for colNum in range(0,len(x)):
        temp = abs(x[colNum] - y[colNum]) ** p
        sum = sum + temp
    dist = sum ** (1.0/p)
    return dist

# finds the mean of the column (feature_num)
def find_mean(cluster, feature_num):
    # some clusters are empty?
    if(cluster.shape[0] == 0):
        return 0.0;
    sum = 0
    for data_points in cluster:
        sum = sum + data_points[feature_num]
    return sum / cluster.shape[0]

def k_means_cs171(x_input, K, init_centroids):
    cluster_assignment = np.arange(x_input.shape[0])

    # iterate through every data_point in x_input
    # find the distances between the init init_centroids
    # push the cluster number (index) of the smallest (closest) distance
    for x in range(0, x_input.shape[0]):
        dist = np.arange(init_centroids.shape[0], dtype=float)
        for y in range(0, init_centroids.shape[0]):
            dist[y] = distance(x_input[x], init_centroids[y], 2)
        cluster_assignment[x] = np.argmin(dist) # returns index of minimum distance

    list_cluster_i = []  # dims are (#data points in cluster # features)
    for x in range(0, K):
        list_cluster_i.append([])

    for x in range(0, cluster_assignment.shape[0]):
        list_cluster_i[cluster_assignment[x]].append(x_input[x])

    # trying to convert it from 2d list to np array
    cluster_i = []
    for x in range(0, len(list_cluster_i)):
        cluster_i.append(np.array(list_cluster_i[x]))
    cluster_i = np.array(cluster_i)

    cluster_centroids = []  # dims are (k x features;)
    # now we want to create the centroid w/ means of respective clusters
    for x in range(0, cluster_i.shape[0]):
        temp_centroid = np.arange(x_input.shape[1], dtype=float)
        for y in range(0, x_input.shape[1]):
            temp_centroid[y] = find_mean(cluster_i[x], y)
        cluster_centroids.append(temp_centroid)
    cluster_centroids = np.array(cluster_centroids)

    return (cluster_assignment, cluster_centroids)

def main():
    dataSet = np.loadtxt('iris.data.txt', delimiter=',', usecols=(0, 1, 2, 3))
    k_clusters = input("How many clusters? ")
    init_centroids = []

    # init_centroids should be random data points
    # so instead of shuffling dataSet and losing track, we should index
    random_arr_for_init_centroids = np.arange(dataSet.shape[0])   # [0 - 149]
    np.random.shuffle(random_arr_for_init_centroids);   #[shuffle]
    # take first k of randomly shuffled
    for x in range(0, k_clusters):
        init_centroids.append(dataSet[random_arr_for_init_centroids[x]])
    init_centroids = np.array(init_centroids);  # convert to numpy array to be easier to work with

    output_tuple = k_means_cs171(dataSet, k_clusters, init_centroids)
    print output_tuple[0]
    print output_tuple[1]

if __name__ == "__main__":
    main()