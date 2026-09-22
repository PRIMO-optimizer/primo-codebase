.._clustering_method:

Clustering Methodology
======================

In PRIMO, the Agglomerative Clustering method is utilized to classify wells into clusters based on their 
characteristics. Currently, PRIMO considers location, age, and depth as the primary characteristics for 
clustering wells. This method groups sample data by calculating linkage distances between data points.
PRIMO employs the scikit-learn Agglomerative Clustering package; more information regarding the method 
can be found at `Agglomerative Clustering <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>`_.

Before clustering, PRIMO pre-computes a distance matrix among all well candidates, 
which serves as input for the agglomerative clustering process. By setting a distance threshold, 
the number of clusters will be determined during the clustering step. This approach helps reduce computation 
time of the optimization step.

Please see :doc:`../Utilities/clustering_utils` for more details on the clustering utilities.