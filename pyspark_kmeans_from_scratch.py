import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import tqdm
import sys


import findspark
findspark.init('/usr/local/Cellar/apache-spark/3.0.1/libexec')


from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession


conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf = conf)
spark = SparkSession(sc)


def compute_euclidian_distance(first, second):
    return math.sqrt((first[0] - second[0])**2 + (first[1] - second[1])**2)



def establish_initial_centroids_index(centroids_number, index_column):
    centroid_indexes = []
    while len(centroid_indexes) < 3:
        current = random.choice(list(index_column))
        while current in centroid_indexes:
            current = random.choice(list(index_column))
        centroid_indexes.append(current)
    return centroid_indexes


def get_cluster_index_given_point_and_centroids(p, centroids):
    closest_centroid = 0
    smallest_distance = None
    for i in range(len(centroids)):
        current_distance = compute_euclidian_distance(p, centroids[i])
        if smallest_distance == None:
            closest_centroid = i
            smallest_distance = current_distance
        if current_distance < smallest_distance:
            closest_centroid = i
            smallest_distance = current_distance
    return closest_centroid


def compute_coordinates_new_centroid(points_list):
    x = []
    y = []
    for point in points_list:
        x.append(point[0])
        y.append(point[1])
    return (sum(x) / len(x), sum(y) / len(y))


def get_new_centroid_coordinates(rdd):
    return rdd.groupByKey()\
              .map(lambda cluster_points_list: (cluster_points_list[0], 
                                      compute_coordinates_new_centroid(cluster_points_list[1])))\
              .map(lambda cluster_centroid_coordinates: cluster_centroid_coordinates[1])\
              .collect()

def compute_variance_for_given_cluster(cluster_point, list_of_points):
    return sum([compute_euclidian_distance(cluster_point, point) for point in list_of_points])


def get_variance_within_each_cluster(rdd, current_configuration):
    return rdd.groupByKey()\
              .map(lambda cluster_index_points_list: (cluster_index_points_list[0],
                          compute_variance_for_given_cluster(current_configuration[cluster_index_points_list[0]],
                                                            cluster_index_points_list[1])))\
              .collect()


def K_means(rdd, k, initial_centroids_indexes, verbose=False, output_file=None):

    # get the coordinates for the initial centroids
    initial_centroids_coordinates = rdd.zipWithIndex()\
                                        .filter(lambda point_index: point_index[1] in initial_centroids_indexes)\
                                        .map(lambda point_index: point_index[0])\
                                        .collect()

    # modify the rdd so that each point is assigned to the closest centroid from the initial chosen ones
    rdd = rdd.map(lambda point: (get_cluster_index_given_point_and_centroids(point, current_configuration), point))


    # iterate until we encounter the same K-configuration, and implicitly the same K-partition as the one from the 
    # previous step

    current_configuration = initial_centroids_coordinates
    prev_configuration = None
    iteration = 0

    while current_configuration != prev_configuration:
        prev_configuration = current_configuration

        rdd = rdd.map(lambda cluster_point: \
                        (get_cluster_index_given_point_and_centroids(cluster_point[1], prev_configuration), 
                         cluster_point[1]))

        current_configuration = get_new_centroid_coordinates(rdd)

        iteration += 1

        if verbose:
        	rdd.saveAsTextFile(output_file + "/iteration" + str(iteration) + ".txt")
        	print('Previous configuration', prev_configuration)
        	print('Next configuration', current_configuration)
        	print(rdd.collect())
        	print('\n\n')

    variance_within_clusters = get_variance_within_each_cluster(rdd, current_configuration)
    variance_summed = sum([cluster_variance[1] for cluster_variance in variance_within_clusters])
    
    if verbose:
        # convert the rdd to a pandas dataframe and visualize the results in a scatter plot
        pd_df = rdd.map(lambda cluster_point: [cluster_point[1][0], cluster_point[1][1], cluster_point[0]])\
                   .toDF(["x", "y", "cluster_index"]).toPandas()
        
        print(pd_df.head())
        print('\n\n')

        plt.scatter(pd_df['x'], pd_df['y'], c=pd_df['cluster_index'])
        plt.show()
        
    return variance_summed


def run_optimized_k_means(filename, k, stopping_iterations):
    # read data and cast from string to numbers
    rdd =  sc.textFile(filename)\
            .map(lambda x: str(x)) \
            .map(lambda w: w.split(',')) \
            .map(lambda point: (eval(point[0]), eval(point[1])))

    iteration = 0
    best_initial_clusters_indexes = None
    min_variance = None

    while iteration < stopping_iterations or current_variance == stopping_variance:

        # establish randomly the indexes for the first K centroids
        initial_centroids_indexes = establish_initial_centroids_index(K, range(rdd.count()))
        current_variance = K_means(rdd, K, initial_centroids_indexes)

        if min_variance == None:
            best_initial_clusters_indexes = initial_centroids_indexes
            min_variance = current_variance
        elif min_variance > current_variance:
            print("Improved variance at iteration = " + str(iteration) + ' with the initial cluster indexes ' +
                  str(initial_centroids_indexes))
            best_initial_clusters_indexes = initial_centroids_indexes
            min_variance = current_variance

        iteration += 1
    
    return best_initial_clusters_indexes


def run_unoptimized_kmeans(filename, k, verbose=False):
    # read data and cast from string to numbers
    rdd =  sc.textFile(filename)\
            .map(lambda x: str(x)) \
            .map(lambda w: w.split(',')) \
            .map(lambda point: (eval(point[0]), eval(point[1])))
    
    # establish randomly the indexes for the first K centroids
    initial_centroids_indexes = establish_initial_centroids_index(K, range(rdd.count()))
    K_means(rdd, K, best_initial_clusters_indexes, verbose)


K=3

filename_input = sys.argv[1]
output_file = sys.argv[2]
stopping_variance = 1.0
stopping_iterations = 10

best_initial_clusters_indexes = run_optimized_k_means(filename_input, K, stopping_iterations)

rdd = sc.textFile(filename_input)\
            .map(lambda x: str(x)) \
            .map(lambda w: w.split(',')) \
            .map(lambda point: (eval(point[0]), eval(point[1])))

K_means(rdd, K, best_initial_clusters_indexes, verbose=True, output_file=output_file)
