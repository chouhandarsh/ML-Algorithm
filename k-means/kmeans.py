import random
import numpy as np
print()
class KMeans:
    def __init__(self,n_clusters=2,max_itr=100):
        self.n_clusters=n_clusters
        self.max_itr= max_itr
        self.centroids=None
    def fit_predict(self,X):
        random_index=random.sample(range(0,X.shape[0]),self.n_clusters)
        self.centroids = X[random_index]
        # now we will interate for max iterations and then we will assign clusters move centyroids check finish
        for i in range(1,self.max_itr):
            cluster_grp = self.assign_clusters(X)
            old_centroid = self .centroids
            self.centroids = self.move_centroids(X,cluster_grp)
            if(self.centroids==old_centroid).all():
                break
        return cluster_grp
    def assign_clusters(self,X):
        cluster_grp=[]
        
        for row in X:
            distances=[]
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distances  = min(distances)
            index_pos = distances.index(min_distances)
            cluster_grp.append(index_pos)
        return np.array(cluster_grp)
    def move_centroids(self,X,cluster_grp):
        # so we will find the mean of all the points that belong to the same cluster and that is going to be my new 
        # no of clusters
        new_centroid=[]
        cluster_type=np.unique(cluster_grp)
        for type in cluster_type:
            new_centroid.append((X[cluster_grp == type]).mean(axis=0))
        return np.array(new_centroid) 

        

