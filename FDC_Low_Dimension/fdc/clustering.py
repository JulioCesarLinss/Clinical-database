import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from fdc.visualize import plotCluster
from hdbscan import HDBSCAN

class Clustering:
    def __init__(self, high_dim, low_dim, visual):
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.visual = visual

    def Agglomerative(self, number_of_clusters, metric, linkage):
        ag_cluster = AgglomerativeClustering(n_clusters=number_of_clusters, metric=metric, linkage=linkage)
        clusters = ag_cluster.fit_predict(self.high_dim)
        (values, counts) = np.unique(clusters, return_counts=True)
        self.low_dim['Cluster'] = clusters
    
        if self.visual:
            plotCluster(self.low_dim, clusterName="Cluster", xName="FDC_0", yName="FDC_1", stroke=3)

        return self.low_dim.Cluster.to_list(), counts
    
    
    def DBSCAN(self, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(self.high_dim)
        (values, counts) = np.unique(clusters, return_counts=True)
        self.low_dim['Cluster'] = clusters
    
        if self.visual:
            plotCluster(self.low_dim, clusterName="Cluster", xName="FDC_0", yName="FDC_1", stroke=3)

        return self.low_dim.Cluster.to_list(), counts
    
    def K_means(self, no_of_clusters):
        kmeans = KMeans(n_clusters=no_of_clusters)
        clusters = kmeans.fit_predict(self.high_dim)
        (values, counts) = np.unique(clusters, return_counts=True)
        self.low_dim['Cluster'] = clusters
    
        if self.visual:
            plotCluster(self.low_dim, clusterName="Cluster", xName="FDC_0", yName="FDC_1", stroke=3)

        return self.low_dim.Cluster.to_list(), counts

#algorimto hdbscan
    def HDBSCAN(self, min_cluster_size=15):
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        clusters = clusterer.fit_predict(self.high_dim)
        noise = np.sum(clusters == -1)
        (values, counts) = np.unique(clusters, return_counts=True)
        print(f"Pacientes perdidos como ruído: {noise} de {len(clusters)}")
        print(f"Clusters encontrados: {len(values[values >= 0])}")
        self.low_dim['Cluster'] = clusters
        if self.visual:
            plotCluster(self.low_dim, clusterName="Cluster", xName="FDC_0", yName="FDC_1", stroke=3)
        return self.low_dim.Cluster.to_list(), counts