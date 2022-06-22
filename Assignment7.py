import random
import numpy as np

# Generate arrays with all the data through numpy

table_general = np.loadtxt("zoo_2.txt", dtype = str)
length_column = table_general.shape[1]
length_row = table_general.shape[0]
table_species = np.loadtxt("zoo_2.txt", dtype = str, skiprows = 1, usecols = 0)
table_entry = np.loadtxt("zoo_2.txt", dtype = int, skiprows = 1, usecols =
range(1, length_column))
table_attribute = np.loadtxt("zoo_2.txt", dtype = str, max_rows = 1, usecols = range(1, length_column))

# Part 2
# Manhattan Metric

cluster_dicts_lists = []

cluster_centers = np.ones(7)

for i in range(7):
    cluster_centers[i] = random.randint(1, 100)

dict_cluster = {}
for i in range(1,8):
    dict_cluster[i] = [[], np.zeros(15)]

for i in range(1, length_row): # Manhattan Metrics between animal and cluster animal
    array_manhattan = np.zeros(7)
    for j in range(len(cluster_centers)):
        if (int(cluster_centers[j])!=i):
            manhattan_sum = 0
            for m in range(1, length_column):
                manhattan_sum+=abs(int(table_general[i][m]) -
                                   int(table_general[int(cluster_centers[j])][m]))
            array_manhattan[j]= manhattan_sum

    smallest = array_manhattan[1]
    index_small = 1
    for n in range(len(array_manhattan)):
        if (array_manhattan[n]<smallest) :
            smallest = array_manhattan[n]
            index_small = n

    dict_cluster[index_small + 1][0].append(table_species[i - 1])
    dict_cluster[index_small + 1][1]+= table_entry[i - 1]


for i in dict_cluster:
    if (len(dict_cluster[i][0])>0):
        dict_cluster[i][1]= dict_cluster[i][1] / float(len(dict_cluster[i][0]))

cluster_dicts_lists.append(dict_cluster)

dict_cluster = {}
for i in range(1,8):
     dict_cluster[i] = [[], np.zeros(15)]

for i in range(99):
    for j in range(1, length_row):
        array_manhattan = np.zeros(7)
        for k in range(1,8):
            manhattan_sum = 0
            for m in range(1, length_column):
                manhattan_sum+=abs(int(table_general[j][m]) -
                                   cluster_dicts_lists[i][k][1][m - 1])
        array_manhattan[k - 1] = manhattan_sum

    smallest = array_manhattan[1]
    index_small = 1
    for n in range(len(array_manhattan)):
        if (array_manhattan[n]<smallest):
            smallest = array_manhattan[n]
            index_small = n
    dict_cluster[index_small + 1][0].append(table_species[j - 1])
dict_cluster[index_small + 1][1]+= table_entry[j - 1]

for i in dict_cluster:
    if (len(dict_cluster[i][0])>0):
        dict_cluster[i][1] = dict_cluster[i][1] / float(len(dict_cluster[i][0]))

 cluster_dicts_lists.append(dict_cluster)

dict_cluster = {}
for i in range(1,8):
     dict_cluster[i] = [[], np.zeros(15)]

# Display & Comparison to Right Answer For Manhattan

print("\nManhattan Results\n")

jaccard_array_euclidean = np.zeros(7)

for i in range(1,8):
    jaccard_sim = 0.0
    intersect = 0
    print("Cluster ", i,": ", cluster_dicts_lists[len(cluster_dicts_lists) - 1][i][0])
    right_answer_names = np.loadtxt("class_2.txt", dtype = str, skiprows = i, max_rows = 1)

    for j in range(len(cluster_dicts_lists[len(cluster_dicts_lists) - 1][i][0])):
        for k in range(2, len(right_answer_names)):
            if (cluster_dicts_lists[len(cluster_dicts_lists) - 1][i][0]

[j]== right_answer_names[k]):
                    intersect+=1

    jaccard_sim = intersect /
(len(cluster_dicts_lists[len(cluster_dicts_lists) - 1][i][0]) +
(len(right_answer_names) - 2))
    jaccard_array_euclidean[i - 1]+= jaccard_sim


# Euclidean Distance

cluster_dicts_lists = []

cluster_centers = np.ones(7)

for i in range(7):
    cluster_centers[i] = random.randint(1, 100)

dict_cluster = {}
for i in range(1,8):
    dict_cluster[i] = [[], np.zeros(15)]

for i in range(1, length_row):
    euclid_array = np.zeros(7)
    for j in range(len(cluster_centers)):
        if (int(cluster_centers[j])!=i):
        sum_euclid = 0
        for m in range(1, length_column):
            sum_euclid+= (int(table_general[i][m]) -
                          int(table_general[int(cluster_centers[j])][m])) ** 2
            sum_euclid = sum_euclid**0.5
            euclid_array[j]= sum_euclid

    smallest = euclid_array[1]
    index_small = 1
for n in range(len(euclid_array)):
    if (euclid_array[n]<smallest):
        smallest = euclid_array[n]
        index_small = n

dict_cluster[index_small + 1][0].append(table_species[i - 1])
dict_cluster[index_small + 1][1]+= table_entry[i - 1]

for i in dict_cluster:
    if (len(dict_cluster[i][0])>0):
    dict_cluster[i][1]= dict_cluster[i][1] / float(len(dict_cluster[i][0]))

cluster_dicts_lists.append(dict_cluster)

dict_cluster = {}
for i in range(1,8):
    dict_cluster[i] = [[], np.zeros(15)]

for i in range(99):
    for j in range(1, length_row):
        euclid_array = np.zeros(7)
        for k in range(1,8):
             sum_euclid = 0
             for m in range(1, length_column):
                sum_euclid+= (int(table_general[j][m]) - cluster_dicts_lists[i][k][1][m - 1]) ** 2

             sum_euclid = sum_euclid**0.5
             euclid_array[k-1] = sum_euclid

             smallest = euclid_array[1]
             index_small = 1

             for n in range(len(euclid_array)):
                if (euclid_array[n]<smallest):
                    smallest = euclid_array[n]
                    index_small = n

 dict_cluster[index_small + 1][0].append(table_species[j - 1])
 dict_cluster[index_small + 1][1]+= table_entry[j - 1]

 for i in dict_cluster:
     if (len(dict_cluster[i][0])>0):
         dict_cluster[i][1] = dict_cluster[i][1] / float(len(dict_cluster[i][0]))

 cluster_dicts_lists.append(dict_cluster)

 dict_cluster = {}
 for i in range(1,8):
     dict_cluster[i] = [[], np.zeros(15)]

# Display & Comparison to Right Answer For Euclidean

print("\nEuclidean Results\n")

jaccard_euclidean = np.zeros(7)
for i in range(1,8):
    jaccard_sim = 0.0
    intersect = 0
    print("Cluster ", i,": ", cluster_dicts_lists[len(cluster_dicts_lists) - 1][i][0])

    right_answer_names = np.loadtxt("class_2.txt", dtype = str, skiprows = i, max_rows = 1)

    for j in range(len(cluster_dicts_lists[len(cluster_dicts_lists) - 1][i][0])):
        for k in range(2, len(right_answer_names)):
            if(cluster_dicts_lists[len(cluster_dicts_lists) - 1][i][0]
[j]==right_answer_names[k]):
               intersect+=1

    jaccard_sim = intersect /
(len(cluster_dicts_lists[len(cluster_dicts_lists) - 1][i][0]) +
 (len(right_answer_names) - 2))
    jaccard_euclidean[i - 1]+=jaccard_sim

# Compare Euclidean and Manhattan ---------------------

manhattan_results = np.sum(jaccard_array_euclidean) / len(jaccard_array_euclidean)
euclid_results = np.sum(jaccard_euclidean) / len(jaccard_euclidean)
print("The similarity between right answer and manhattan metric is: ", manhattan_results)
print("The similarity between right answer and euclidean distance is: ", euclid_results)
print("I found through multiple tests of running the program that the Manhattan "
      "Metric tends to perform better in terms of similarity to the right answer")
