from collections import defaultdict
import math
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

# Cluste size.
K = 5
#MAX_ITER = 3;

# List of cluster with its points in it.
CLUSTER = defaultdict(list);

# Kmean model.
model = KMeans(n_clusters=K, max_iter=300)


# Anomoly threshold. Need to be tuned to avoid over / under fitting.
Thres = 0.95

# Data frames loaded from csv.
df = pd.read_csv('C:\\Users\\himverma\\AnacondaProjects\\KMeans\\xclaraOriginal.csv')
#print(df.describe())
#print(df.shape)

def distance(x, y):
    """
    Find distance between two points in a plain.
    @param x: 2D point.
    @param y: 2D point.
    
    @return euclidean distance between this point.
    """
    d1 = x[0] - y[0];
    d2 = x[1] - y[1];
    distance = math.sqrt(d1*d1 + d2*d2)
    return distance


def print_cluster_details(clusters, centroids):
    for index, cluster in clusters.items():
        print ("Cluster: {} size: {}".format(index, len(cluster)))

        
def dump_cluster_points(df, labels):
    """
    @param clusters: dataframe
    
    Dump points of the cluster in csv file named as cluster_{#index}.csv
    """
    clusters = aggregate_cluster_points(df, labels)
    for index, cluster in clusters.items():
        with open("C:\\Users\\himverma\\AnacondaProjects\\KMeans\\cluster_{}.csv".format(index), "w") as f:
            f.write("\n".join(["{},{}".format(p[0], p[1]) for p in cluster]))
            
def aggregate_cluster_points(df, labels):
    """
    Helper methods to aggregate the cluster points based on the label index.
    
    @param df: List of points or datapoints
    @param labels: Cluster index list for each element in points.

    @return List of cluster points, indexed with cluster index.
    """
    clusters = defaultdict(list)
    #print("labels",labels)
    for index, value in enumerate(labels):
        clusters[value].append(df.values[index])
   # print ("clusters",clusters)
    return clusters
    

def get_outliers_and_strip_cluster(cluster_points, centroid):
    """
    Apply ODIN algorithm to identify anomalies in the cluster and
    strip it.
    
    Anomaly detection rule:- 
    
    sqrt(point^2 - centroid^2) / max(points) > T === True then it's an anomaly.
    
    @param cluster_points: List of points in this cluster.
    @param centroid: centroid of the cluster.
    @return: outliers, new_cluster
    """
    #print("clusterPoints",cluster_points)
    d_vector = np.array([distance(point, centroid)
                         for point in cluster_points])
    d_max = d_vector.max();
    #print("d_max",d_max)
    data = pd.DataFrame([distance(centroid, point) / d_max
                         for point in cluster_points])
    #print("data",data)
#     print data.min(), d_max
    outliers = filter(lambda row: distance(centroid, row) / d_max > Thres,
                      cluster_points)
    new_cluster = filter(lambda row: distance(centroid, row) / d_max <= Thres,
                         cluster_points)
    
   # print ("cluster_points",cluster_points)
    #print("centroid",centroid)
    #for i in cluster_points:
        #print("distance",distance(centroid,i)/d_max)
        #print(distance(centroid, row) / d_max)
        #print ("Original cluster size: ",len(cluster_points))
    return outliers, new_cluster


def run_outlier_removal_clustering(df, max_iteration):
    """
    Run ORC Outlier removal clustering on the datapoints. 
    Clustering Algorithm - KMean
    Outlier removal Algorithm - ODIN a Knn based outlier detection.
    """
    orc_model = KMeans(n_clusters=K, max_iter=MAX_ITER)
    OUTLIERS = []
    for iteration in range(max_iteration):
        # Iteration.
        #print "\n\n[{}] ===> Data before clustering: {}, Anomaly: {}".format(
        #iteration, df.shape, len(OUTLIERS))
        orc_model.fit(df)
        labels = orc_model.labels_

        CLUSTER = aggregate_cluster_points(df, labels)
        centroids = orc_model.cluster_centers_
        
        NEW_CLUSTER = []
        for index, cluster in CLUSTER.items():
            #print ("Cluster: {} size: {}".format(index, len(cluster)))
            outlier, new_cluster = get_outliers_and_strip_cluster(cluster,
                                                                  centroids[index])

            OUTLIERS.extend(outlier)
            NEW_CLUSTER.extend(new_cluster)
            #print("outliers",OUTLIERS)
            #print("new_cluster",NEW_CLUSTER)
        # Update the cluster with new cluster.
        df = pd.DataFrame(data=NEW_CLUSTER)
        
    # Fit for the one more time, as the when loop exists we removed few anomolies.
    orc_model.fit(df)

    return df, orc_model, OUTLIERS
    
    
# Run Clustering with Outlier removal algorithm.
df, orc_model, outliers = run_outlier_removal_clustering(df, 1)



# Dump the final cluster and anomalies into csv file.
print_cluster_details(aggregate_cluster_points(df, orc_model.labels_),
                      orc_model.cluster_centers_)
print ("Total anomalies: {}".format(len(outliers)))
print ("Exported the cluster and anomalies into csv files")
dump_cluster_points(df, orc_model.labels_)
with open("C:\\Users\\himverma\\AnacondaProjects\\KMeans\\anomalies.csv", 'w') as f:
    f.write("\n".join(["{},{}".format(p[0], p[1]) for p in outliers]))
    

# Plot the Original and new cluster after anomaly removal.
plt.figure(figsize=(12,4))
colormap = np.array(['red', 'lime', 'blue', 'green', 'yellow'])
df.columns = ['x', 'y']

data = pd.read_csv('C:\\Users\\himverma\\AnacondaProjects\\KMeans\\xclaraOriginal.csv')
plt.subplot(1, 3, 1)
plt.scatter(data.V1, data.V2, s=20)
plt.title("Without clustering")

plt.subplot(1, 3, 2)
_kmean = model.fit(data)
plt.scatter(data.V1, data.V2, c=colormap[_kmean.labels_], s=20)
plt.title("KMean Clustering")

plt.subplot(1, 3, 3)
plt.scatter(df.x, df.y, c=colormap[orc_model.labels_], s=20)
plt.title("ORC clustering")