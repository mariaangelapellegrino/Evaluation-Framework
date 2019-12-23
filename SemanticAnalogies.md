## Semantic Analogies

### Datasets used as gold standard

| **Dataset** | **Structure** | **Size** | **Source** |
| :---------: | :---------------------: | : ----------: |  : ----------: | 
| Capitals and countries | ca1 co1 ca2 co2 | 505 | [Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) |
| Currency (and Countries)| cu1 co1 cu2 co2 | 866 | [Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) |
| Cities and State | ci1 st1 ci2 st2 | 2,467 | [Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) |
| (All) capitals and countries | ca1 co1 ca2 co2| 4,523 | [Word2Vec](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) |

### Model 
The task takes the quadruplets (v\_1, v\_2, v\_3, v\_4) and works on the first three vectors to predict the fourth one.
Among all the vectors, the nearest to the predicted one is retrieved, where the closest vector is computed by the dot product. 

      def default_analogy_function(a, b, c){ return b - a + c }

The vector returned by the function (the _predicted vector_) gets compared with the _top\_k_ most similar ones. 
If the actual forth vector is among the _top\_k_ most similar ones, the answer is considered correct. 

The analogy function to compute the predicted vector and the top\_k value can be customised. 

### Output of the evaluation

| **Metric** | **Range** | **Interpretation** |
| :---------: | :---------------------: | :----------: |
| Accuracy | \[0,1\] | Highest |
