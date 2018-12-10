import pandas as pd

class DataManager:
    def __init__(self):
        print("Clustering data manager initiated")

    @staticmethod
    def read_data(vectors, gold_file, column_id=None, column_cluster=None):
        if column_id is None:
            column_id = 'DBpedia_URI'
        if column_cluster is None:
            column_cluster = 'cluster'

        fields = [column_id, column_cluster]
        
        gold = pd.read_csv(gold_file, usecols=fields, delim_whitespace=True, index_col=False, header=None, names=fields, skipinitialspace=True, skip_blank_lines=True, encoding='utf-8')

        gold.rename(columns={column_id: 'name'}, inplace=True)
        gold.rename(columns={column_cluster: 'cluster'}, inplace=True)

        merged = pd.merge(gold, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(gold, vectors, how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

        return (merged, ignored)
