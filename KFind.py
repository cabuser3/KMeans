import pandas as pd
import pylab as pl
import numpy as np
from sklearn.cluster import KMeans

dF = pd.read_csv('C:\\Users\\himverma\\AnacondaProjects\\KMeans\\data.csv')
print("Input Data and Shape")
print(dF.shape)
print(list(dF))
#print(dF)
# Getting the values and plotting it
#f1 = data['V1'].values
#f2 = data['V2'].values
#X = np.array(list(zip(f1, f2)))

Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(dF).score(dF) for i in range(len(kmeans))]
score
pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()