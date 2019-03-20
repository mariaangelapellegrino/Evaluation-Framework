from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
import numpy as np

from code.abstract_model import AbstractModel

class ClusteringModel(AbstractModel):

    def __init__(self, modelName, metric, n_clusters, debugging_mode):
        self.n_clusters = n_clusters
        self.debugging_mode = debugging_mode
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
        else:
            print("YOU CHOSE WRONG MODEL FOR CLUSTERING!")
            
        if self.debugging_mode:
            print('Clustering model initialized')

    def train(self, merged, ignored):
        n_samples = merged.shape[0]
        if n_samples < self.n_clusters:
            raise ValueError(
                ("Clustering : Cannot have number of cluster n_clusters={0} greater"
                 " than the number of samples: {1}.").format(self.n_clusters, n_samples) + "\n")
        
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

        if self.debugging_mode:
            print(self.name + ' Number of clusters : ' + str(n_clusters))

        trueLabels = merged.iloc[:, 1]
        trueLabels = np.concatenate((trueLabels, ignored['cluster']), axis=0)  
                
        if labels.ndim!=1:
            raise ValueError(
                "predicted labels must be 1D: shape is %r" % (labels.shape,))
        if trueLabels.ndim!=1:
            raise ValueError(
                "true labels must be 1D: shape is %r" % (trueLabels.shape,))
        if labels.shape != trueLabels.shape:
            raise ValueError(
                "true labels and predicted labels must have same size, got %d and %d"
                % (trueLabels.shape[0], labels.shape[0]))      

        adjusted_rand_score = metrics.adjusted_rand_score(trueLabels, labels)
        if self.debugging_mode:
            print(self.name + ' Adjusted rand index : ' + str(adjusted_rand_score))

        adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(trueLabels, labels)
        if self.debugging_mode:
            print(self.name + ' Adjusted mutual info score : ' + str(adjusted_mutual_info_score))

        fowlkes_mallows_score = metrics.fowlkes_mallows_score(trueLabels, labels)
        if self.debugging_mode:
            print(self.name + ' Fowlkes_mallows_score : ' + str(fowlkes_mallows_score))

        homogeneity_score = metrics.homogeneity_score(trueLabels, labels) 
        if self.debugging_mode:
            print(self.name + ' Homogeneity_score : ' + str(homogeneity_score))

        completeness_score = metrics.completeness_score(trueLabels, labels) 
        if self.debugging_mode:
            print(self.name + ' Completeness_score : ' + str(completeness_score))

        v_measure_score = metrics.v_measure_score(trueLabels, labels) 
        if self.debugging_mode:
            print(self.name + ' V_measure_score : ' + str(v_measure_score))

        return {'task_name':'Clustering', 'model_name':self.name, 'model_configuration':self.configuration, 'num_clusters' :n_clusters, 
            'adjusted_rand_index':adjusted_rand_score, 'adjusted_mutual_info_score':adjusted_mutual_info_score, 
            'fowlkes_mallows_score':fowlkes_mallows_score, 'homogeneity_score':homogeneity_score, 
            'completeness_score':completeness_score, 'v_measure_score': v_measure_score }          