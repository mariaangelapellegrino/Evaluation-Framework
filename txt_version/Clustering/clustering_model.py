from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np


class Model:

    def __init__(self, modelName, metric=None, n_clusters=None):
        if metric is None:
            metric = "cosine"
        if n_clusters is None:
            n_clusters = 5

        if modelName == "DB":
            self.model = "DBSCAN(metric='"+metric+"')"
            self.name = "DBSCAN"
            self.configuration = "metric="+metric
        elif modelName == "KMeans":
            self.model = "KMeans(n_clusters="+str(n_clusters)+", random_state=0)"
            self.name = "KMeans"
            self.configuration = "metric=enclidean, n_clusters="+str(n_clusters)
        elif modelName == "AC":
            self.model = "AgglomerativeClustering(n_clusters="+str(n_clusters)+", affinity='"+metric+"', linkage='average')"
            self.name = "Agglomerative clustering"
            self.configuration = "metric="+metric+ ", n_clusters="+str(n_clusters)
        elif modelName == "WHC":
            self.model = "AgglomerativeClustering(n_clusters="+str(n_clusters)+", affinity='euclidean', linkage='ward')"
            self.name = "Ward hierarchical clustering"
            self.configuration = "metric="+metric+ ", n_clusters="+str(n_clusters)
        elif modelName == "SC":
            self.model = "SpectralClustering(n_clusters="+str(n_clusters)+", affinity='"+metric+"')"
            self.name = "SpectralClustering clustering"
            self.configuration = "metric="+metric+ ", n_clusters="+str(n_clusters)
        else:
            print("YOU CHOSE WRONG MODEL FOR CLUSTERING!")

    def train(self, merged, ignored):
        data = merged.iloc[:, 2:].values
        data = StandardScaler().fit_transform(data)
        
        cluster_method = eval(self.model).fit(data)
        
        labels = cluster_method.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
               
        for index, value in enumerate(labels):
            if value == -1:
                labels[index] = n_clusters
                n_clusters += 1
                
        ignoredLabels = [n_clusters] * len(ignored)
        labels = np.concatenate((labels, ignoredLabels), axis=0)

        print(self.name + ' Number of clusters : ' + str(n_clusters))

        trueLabels = merged.iloc[:, 1]
        trueLabels = np.concatenate((trueLabels, ignored['cluster']), axis=0)        

        adjusted_rand_score = metrics.adjusted_rand_score(trueLabels, labels)
        print(self.name + ' Adjusted rand index : ' + str(adjusted_rand_score))

        adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(trueLabels, labels)
        print(self.name + ' Adjusted mutual info score : ' + str(adjusted_mutual_info_score))

        fowlkes_mallows_score = metrics.fowlkes_mallows_score(trueLabels, labels)
        print(self.name + ' Fowlkes_mallows_score : ' + str(fowlkes_mallows_score))

        homogeneity_score = metrics.homogeneity_score(trueLabels, labels) 
        print(self.name + ' Homogeneity_score : ' + str(homogeneity_score))

        completeness_score = metrics.completeness_score(trueLabels, labels) 
        print(self.name + ' Completeness_score : ' + str(completeness_score))

        v_measure_score = metrics.v_measure_score(trueLabels, labels) 
        print(self.name + ' V_measure_score : ' + str(v_measure_score))

        return {'task_name':'Clustering', 'model_name':self.name, 'model_configuration':self.configuration, 'num_clusters' :n_clusters, 
            'adjusted_rand_index':adjusted_rand_score, 'adjusted_mutual_info_score':adjusted_mutual_info_score, 
            'fowlkes_mallows_score':fowlkes_mallows_score, 'homogeneity_score':homogeneity_score, 
            'completeness_score':completeness_score, 'v_measure_score': v_measure_score }          
