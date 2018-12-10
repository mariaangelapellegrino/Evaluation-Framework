import pandas as pd

class DataManager:
    def __init__(self):
        print("Classification and regression data manager initialized")

    @staticmethod
    def read_data(vectors, gold_file, task, column_id=None, column_score=None):
        if column_id is None:
            column_id = 'DBpedia_URI15'
        if column_score is None:
            if task==0:
                column_score = 'label'
            else:
                column_score = 'rating'

        fields = [column_id, column_score]
        
        gold = pd.read_csv(gold_file,"\t", usecols=fields, encoding='utf-8')

        gold.rename(columns={column_id: 'name'}, inplace=True)
        gold.rename(columns={column_score: 'label'}, inplace=True)

        merged = pd.merge(gold, vectors, on='name', how='inner')
        outputLeftMerge = pd.merge(gold, vectors, how='outer', indicator=True)
        ignored = outputLeftMerge[outputLeftMerge['_merge'] == 'left_only']

        return (merged, ignored)
