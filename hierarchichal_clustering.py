#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing  dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using dendrogram method to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, 'ward'))
plt.title('Dendrogram')
plt.xlabel('points')
plt.ylabel('distance between clusters / dissimilarities')

#fitting hierarchichal into dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualising the hierarchichal clustering
plt.scatter(X[y_hc == 0,0], X[y_hc == 0,1], c = 'red', label = 'careful')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1,1], c = 'blue', label = 'standard')
plt.scatter(X[y_hc == 2,0], X[y_hc == 2,1], c = 'magenta', label = 'target')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3,1], c = 'cyan', label = 'careless')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4,1], c = 'green', label = 'sensible')
plt.title('Hierarchichal Clustering')
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (0 - 100)')
plt.legend()
plt.show()