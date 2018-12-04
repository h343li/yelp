import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.cluster import AffinityPropagation, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn import datasets
from sklearn.metrics.pairwise import pairwise_distances
import Kmed

"""
Take command line flag to show visualizations or not
"""


def KMeans_Cluster(np_data, num_clusters):
    """
    Perfrom k-means clustering returns cluster labels.
    """
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(np_data)
    return kmeans.predict(np_data), kmeans.cluster_centers_

def Hierarchical_Cluster(np_data, num_clusters):
    """
    Performs Hierarchical clustering returns only labels.
    """
    h_cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    return h_cluster.fit_predict(np_data)

def Gaussian_Mixture_Model_Clustering(np_data, num_clusters):
    """
    perfroms GMM clustering returns labels.
    """
    gmm = GaussianMixture(n_components=num_clusters, init_params='kmeans')
    gmm.fit(np_data)
    return gmm.predict(np_data), gmm.means_


def Spectrial_Cluster(np_data, num_clusters):
    """
    Performs Spectrial clustering and returns labels.
    **Gives warning via console may not work as expected.
    """
    spectrial = SpectralClustering(n_clusters=num_clusters)
    return spectrial.fit_predict(np_data)

def Birch_Cluster(np_data, num_clusters):
    """
    Perform birch clustering and return labels
    """
    birch = Birch()
    return birch.fit_predict(np_data,num_clusters)

if __name__ == '__main__':
    """Creating Test data, which normally distributed on center(5,5,5), (0,0,0) and (-5,-5,-5)"""

    data1 = np.random.normal(0,1,(100,3))
    data2 = np.random.normal(5,1,(100,3))
    data3 = np.random.normal(-5,1,(100,3))
    data = np.vstack([data1, data2, data3])
    n = len(data)

    num_clusters = 3


    """Perform various clustering methods"""
    kmeans_labels,kmeans_centers = KMeans_Cluster(data, num_clusters)
    print(kmeans_labels)

    hierarchical_labels = Hierarchical_Cluster(data, num_clusters)
    print(hierarchical_labels)

    gmm_labels, gmm_centers = Gaussian_Mixture_Model_Clustering(data, num_clusters)
    print(gmm_labels)

    spectrial_labels = Spectrial_Cluster(data, num_clusters)
    print(spectrial_labels)

    birch_labels = Birch_Cluster(data, num_clusters)
    print(birch_labels)

    D = pairwise_distances(data, metric='euclidean')
    M, C = Kmed.kMedoids(D, 3)
    kmed_labels = np.zeros(n)
    for label in C:
        for point_idx in C[label]:
            kmed_labels[point_idx] = int(label)
    print(kmed_labels)


