## Document Similarity

### Datasets used as gold standard

| **Dataset** | **Structure** | **Size** |
| :---------: | :---------------------: | ----------: | 
|   LP50   | doc1 doc2 avg | 50 docs |

### Model 
The algorithm takes two documents _doc1_ and _doc2_ as its input and calculates their similarity as follows:
- For each document, the related set of entities is retrieved. The output of this step are the sets _E1_ and _E2_, respectively.
- For each pair of entities (i.e. for the cross product of the sets), the similarity score is computed.
- Only the maximum value is preserved for determining the document similarity evaluation. Therefore, for each entity in _E1_ the maximum similarity to an entity in _E2_ is kept and vice versa.
- The similarity score between the two documents is calculated by averaging the sum of all these maximum similarities. 

The _similarity\_function_ can be customized by the user.

The Document Similarity task simply ignores any *missing entities* and computes the similarity only on entities that both occur in the gold standard dataset and in the input file.

### Output of the evaluation

| **Metric** | **Range** | **Interpretation** |
| :---------: | :---------------------: | :----------: |
| Pearson correlation coefficient (P\_cor) | \[-1,1\] | Extreme values: correlation, Values close to 0: no correlation |
| Spearman correlation coefficient (S\_cor) | \[-1,1\] | Extreme values: correlation, Values close to 0: no correlation |
| Harmonic mean of P\_cor and S\_cor  | \[-1,1\] | Extreme values: correlation, Values close to 0: no correlation |
