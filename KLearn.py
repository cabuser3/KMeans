
# coding: utf-8

# In[1]:


# from copy import deepcopy
import numpy as np
import pandas as pd
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib inline')
#%matplotlib inline
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Importing the dataset
data = pd.read_csv('C:\\Users\\himverma\\AnacondaProjects\\KMeans\\xclaraOriginal.csv')
print("Input Data and Shape")
print(data.shape)
data.head()

# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Number of clusters
clusterNumber = 3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X)-20, size=clusterNumber)
#print (C_x)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X)-20, size=clusterNumber)
#print (C_y)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print("Initial Centroids")
print(C)

# Plotting along with the Centroids
plt.scatter(f1, f2, c='#050505', s=8)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
plt.title('Initial Centroids (2D):: ')

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# Loop will run till the error becomes zero
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(clusterNumber):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
plt.title('Logic Results in 2D:: ')
for i in range(clusterNumber):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
        ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')

fig1 = plt.figure()
ax1 = Axes3D(fig1)
plt.title('Logic Results in 3D:: ')
for i in range(clusterNumber):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax1.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
        ax1.scatter(C[:, 0], C[:, 1], marker='*', c='#050505', s=1000)


'''
==========================================================
sk-learn
==========================================================
'''

from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=clusterNumber)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_

# Comparing with scikit-learn centroids
print("Centroid values:: ")
print("From Logic-")
print(C) # From Logic
print("From sklearn-")
print(centroids) # From sci-kit learn

colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig2, ax2 = plt.subplots()
plt.title('sklearn Results:: ')
for i in range(clusterNumber):
            print("coordinate:",X[i], "label:", labels[i])
            points=np.array([X[j] for j in range(len(X)) if labels[j] == i])
            ax2.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
            ax2.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
