## Clustering

### Datasets used as gold standard

| **Dataset** | **Interpretation of clusters** | **Clusters** | **Size** | 
| :---------: | :---------------------: | ----------: | -------: |
|   Teams    |     {Football Teams, Basketball Teams}      |           2 |      4,206 
| (Balanced) Cities and Countries | {Cities, Countries} | 2 | 4,344 |
| Cities, Albums, Movies, AAUP, Forbes | {Cities, Albums, Movies, Universities, Companies}| 5 | 6,357 |
! Cities and Countries | {Cities, Countries| 2 | 11,182 |

### Model and its configuration

| **Model** | **Configuration** |
| :---------: | :---------------------: |
| Agglomerative Clustering | similarity_metric |
| Ward Hierarchical Clustering | similarity_metric |
| DBSCAN | similarity_metric |
| k-Means | - |

For each **missing entity** a *singleton cluster* is created, i.e. a cluster which contains only the current entity. 
Further, soft clustering approaches, such as DBscan, do not cluster all entities. 
We call these entities *miss-clustered entities* and manage them exactly as the missing entities, i.e., we create a singleton cluster for each of them.
The evaluation metrics are applied to the combination of the clusters returned by the clustering algorithm and all the **singleton clusters**.

### Output of the evaluation

| **Metric** | **Range** | **Optimum** |
| :---------: | :---------------------: | ----------: |
| Adjusted rand score | [-1,1] | Highest |
| Adjusted mutual info score | [0,1] | Highest |
| Fowlkes Mallow index | [0,1] | Highest |
| v_measure score| [0,1] | Highest |
| Homogeneity score| [0,1] | Highest |
| Completeness score| [0,1] | Highest |
