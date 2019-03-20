import pandas as pd
import numpy as np
import h5py

class DataManager:
    def __init__(self):
        print("Clustering data manager initiated")

    @staticmethod
    def create_header(vec_size):
        headers = ['name', 'cluster']
        for i in range(0, vec_size):
            headers.append(i)
        return headers

    @staticmethod
    def read_data(vector_filename, vector_size, gold_file, column_id=None, column_cluster=None):
        vector_file = h5py.File(vector_filename, 'r')
        vector_group = vector_file["Vectors"]

        if column_id is None:
            column_id = 'DBpedia_URI_Base32'
        if column_cluster is None:
            column_cluster = 'cluster'

        fields = ['DBpedia_URI', column_id, column_cluster]
        
        gold = pd.read_csv(gold_file, delim_whitespace=True, usecols=fields, index_col=False, skipinitialspace=True, skip_blank_lines=True, encoding='utf-8')

        gold.rename(columns={column_id: 'name'}, inplace=True)
        gold.rename(columns={column_cluster: 'cluster'}, inplace=True)

        merged = pd.DataFrame(columns= DataManager.create_header(vector_size))
        ignored = pd.DataFrame(columns= ['name', 'cluster'])

        for row in gold.itertuples():
            try:
                values = vector_group[row.name][0]
                        
                new_row = dict(zip(np.arange(vector_size), values))
                new_row['name'] = row.name
                new_row['cluster'] = row.cluster

                merged = merged.append(new_row, ignore_index=True)
            except KeyError:
                ignored = ignored.append({'name':row.DBpedia_URI, 'cluster':row.cluster}, ignore_index=True)

        return (merged, ignored)
