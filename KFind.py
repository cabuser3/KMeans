import pandas as pd
import pylab as pl
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = pd.read_csv('C:\\Users\\himverma\\AnacondaProjects\\KMeans\\xclaraOriginal.csv')
print("Input Data and Shape")
print(data.shape)
data.head()

# Getting the values and plotting it
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()