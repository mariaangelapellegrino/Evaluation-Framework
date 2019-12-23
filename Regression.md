## Regression

### Datasets used as gold standard

| **Dataset** | **Semantic of classes** | **Size** | **Source** |
| :---------: | :---------------------: | -------: | :--------: |
|   [Cities](http://data.dws.informatik.uni-mannheim.de/rmlod/LOD_ML_Datasets/data/datasets/CitiesQualityOfLiving/)    |      Living style       |      212 |   Mercer   |
|   [AAUP](http://data.dws.informatik.uni-mannheim.de/rmlod/LOD_ML_Datasets/data/datasets/DatasetsWithFeatures/AAUP/)     |  Salary of professors   |      960 |    JSE     |
|   [Forbes](http://data.dws.informatik.uni-mannheim.de/rmlod/LOD_ML_Datasets/data/datasets/DatasetsWithFeatures/Forbes2013/)    |      Agency income      |    1,585 |   Forbes   |
|   [Albums](http://data.dws.informatik.uni-mannheim.de/rmlod/LOD_ML_Datasets/data/datasets/MetacriticAlbums/)    |    Album popularity     |    1,600 | Metacritic |
|   [Movies](http://data.dws.informatik.uni-mannheim.de/rmlod/LOD_ML_Datasets/data/datasets/MetacriticMovies/)    |    Movie popularity     |    2,000 | Metacritic |

### Model and its configuration

| **Model** | **Configuration** |
| :---------: | :---------------------: |
| Linear Regression | - |
| M5Rules | - |
| k-NN | k=3 |

**Missing entities** are simply ignored.

The results are calculated using stratified _10-fold cross-validation_.

### Output of the evaluation

| **Metric** | **Range** | **Optimum** |
| :---------: | :---------------------: | :----------: |
| Root Mean Squared Error (RMSE) | \[0,1\] | Lowest |
