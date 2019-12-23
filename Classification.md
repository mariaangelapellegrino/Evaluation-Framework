## Classification

### Datasets used as gold standard

| **Dataset** | **Semantic of classes** | **Classes** | **Size** | **Source** |
| :---------: | :---------------------: | ----------: | -------: | :--------: |
|   [Cities](http://data.dws.informatik.uni-mannheim.de/rmlod/LOD_ML_Datasets/data/datasets/CitiesQualityOfLiving/)    |      Living style       |           3 |      212 |   Mercer   |
|   [AAUP](http://data.dws.informatik.uni-mannheim.de/rmlod/LOD_ML_Datasets/data/datasets/DatasetsWithFeatures/AAUP/)     |  Salary of professors   |           3 |      960 |    JSE     |
|   [Forbes](http://data.dws.informatik.uni-mannheim.de/rmlod/LOD_ML_Datasets/data/datasets/DatasetsWithFeatures/Forbes2013/)    |      Agency income      |           3 |    1,585 |   Forbes   |
|   [Albums](http://data.dws.informatik.uni-mannheim.de/rmlod/LOD_ML_Datasets/data/datasets/MetacriticAlbums/)    |    Album popularity     |           2 |    1,600 | Metacritic |
|   [Movies](http://data.dws.informatik.uni-mannheim.de/rmlod/LOD_ML_Datasets/data/datasets/MetacriticMovies/)    |    Movie popularity     |           2 |    2,000 | Metacritic |

### Model and its configuration

| **Model** | **Configuration** |
| :---------: | :---------------------: |
| Naive Bayes | - |
| C 4.5 decision tree | - |
| k-NN | k=3 |
| Support Vector Machine | C = {10^{-3}, 10^{-2}, 0.1, 1, 10, 10^2, 10^3} |

**Missing entities** are simply ignored.

The results are calculated using stratified _10-fold cross-validation_.

### Output of the evaluation

| **Metric** | **Range** | **Optimum** |
| :---------: | :---------------------: | :----------: |
| Accuracy | \[0,1\] | Highest |
