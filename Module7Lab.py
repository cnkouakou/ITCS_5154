from copy import deepcopy as copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import seaborn as sns

from sklearn.datasets import load_iris

# TODO 1
iris = load_iris()
X = iris.data
T = iris.target


print(iris)
print("X (data) shape: {}".format(X.shape))
print("T (labels) shape: {}".format(T.shape))

#http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
# modified by JML

plt.figure(figsize=(12,6))
for i in range(2):
    x_min, x_max = X[:, 2*i].min() - .5, X[:, 2*i].max() + .5
    y_min, y_max = X[:, 2*i+1].min() - .5, X[:, 2*i+1].max() + .5

    plt.subplot(1,2,i+1)
    # Plot the training points
    plt.scatter(X[:, 2*i], X[:, 2*i+1], c=T, cmap=plt.cm.Set1,
                edgecolor='k')
    plt.xlabel('Sepal length' if i==0 else 'Petal length')
    plt.ylabel('Sepal width'  if i==0 else 'Petal width')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
plt.show()

from sklearn.cluster import KMeans

# TODO 2.1-2.2
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
y_pred = kmeans.fit_predict(X)   

# TODO 2-3 (predicted targets)
print("Predicted cluster labels:\n", kmeans.labels_)

# TODO 2-3 (cluster centers)
print("\nCluster centers:\n", kmeans.cluster_centers_)

human_mus  = pd.DataFrame(kmeans.cluster_centers_, columns=iris.feature_names)
human_mus.index = ['cluster {} means'.format(k+1) for k in range(len(human_mus))]
print(human_mus)

def plot_iris_cluster(X, y, mu, dim=2):
    """
        Plot the Itis data with based on passed labels

        Args:
            X (np.ndarray): Data formatted as a NumPy array
            y (np.ndarray): Vector of labels to plot each cluster.
            mu (float): The center of each cluster
            dim (int): option to plot multidimensional figures (for sepal and petal)
    """

    k = len(np.unique(y)) # mu.shape[0]
    plt.figure(figsize=(12,6))
    for i in range(dim):
        x_min, x_max = X[:, 2*i].min() - .5, X[:, 2*i].max() + .5
        y_min, y_max = X[:, 2*i+1].min() - .5, X[:, 2*i+1].max() + .5

        plt.subplot(1,2,i+1)
        # TODO 3.1: Plot the training points
        plt.scatter(X[:, 2*i], X[:, 2*i+1], c=y, cmap=plt.cm.Set1, edgecolors='k')


        # plot the center
        if mu is not None:
            cluster_labels = range(k)
            # TODO 3.2
            plt.scatter(mu[:, 2*i], mu[:, 2*i+1], 
                        c=cluster_labels, cmap="coolwarm", 
                        marker='o', s=200, edgecolors='yellow', label="Centroids")


        plt.xlabel('Sepal length' if i==0 else 'Petal length')
        plt.ylabel('Sepal width'  if i==0 else 'Petal width')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
    plt.show()

# TODO 3.3
plot_iris_cluster(X, y_pred, kmeans.cluster_centers_)

kmeans.set_params(n_clusters = 5)
y_pred = kmeans.fit_predict(X)
plot_iris_cluster(X, y_pred, kmeans.cluster_centers_)

kmeans.set_params(n_clusters = 3)
y_pred = kmeans.fit_predict(X)
plot_iris_cluster(X, y_pred, kmeans.cluster_centers_)
