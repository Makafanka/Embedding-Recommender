import pandas as pd
import numpy as np
from utils import get_embedding
from utils import get_kmeans
from utils import get_tags
from utils import get_visualization
import os.path

import random


class Embedding:
    def __init__(self):
        print("Embedding Class")
    
    def output_embedding(self, df, col_name_desc, new_col_name):
        """
        Get the embeddings for all descriptions and print the 20 most frequently appeared tags
        :param df: pd.dataframe, including pois, decriptions, tags
        :param col_name_desc: str, the name of the column including descriptions
        :param new_col_name: str, the name you want to have for the embeddings column
        :param col_name_em: str, the name of the column including embeddings
        :return embeddings: pd.dataframe, all embeddings
        """
        desc = df[col_name_desc]
        embeddings = []
        for data in desc:
            embeddings.append(get_embedding(data, model="text-embedding-ada-002"))
        embeddings = pd.Series(embeddings)
        
        df[new_col_name] = embeddings

        df.to_csv(os.path.join('../output','em1.csv'), index=False)
        # df.to_csv(os.path.join('../output','embeddings_df.csv'), index=False)
        return df

    def get_evaluation(self, df, col_name_tags, col_name_pois, col_name_em, tag1, tag2, tag3):
        """
        Generate the evaluation for embeddings 
        Note: Need to pick tag1, 2, and 3 from the printed most frequently appeared tags as shown in src/freq_app_tags.txt
        :param df: pd.dataframe, including pois, decriptions, tags, and embeddings
        :param col_name_tags: str, the name of the column including tags
        :param col_name_pois: str, the name of the column including poi names
        :param col_name_em: str, the name of the column including embeddings
        :param tag1: str, the first tag
        :param tag2: str, a tag that is similar to the first tag
        :param tag1: str, a tag that is different from the first tag
        """
        pois = df[col_name_pois]
        tickers = df[col_name_tags]
        embeddings = df[col_name_em]

        matrix = np.array(embeddings.apply(eval).to_list())
        em, cat, cat2, tags = get_tags(matrix, tickers)

        # Generate the visualization 
        get_visualization(tickers, em, tag1, tag2, tag3) 

        # Generate the kmeans clusters
        cluster0, iss = get_kmeans(cat, tags, matrix)
        d0 = cluster0[cluster0.cluster == 0]["data_index"]
        d1 = cluster0[cluster0.cluster == 1]["data_index"]
        d2 = cluster0[cluster0.cluster == 2]["data_index"]
        d3 = cluster0[cluster0.cluster == 3]["data_index"]
        d4 = cluster0[cluster0.cluster == 4]["data_index"]
        table0 = []
        table1 = []
        table2 = []
        table3 = []
        table4 = []
        pois1 = []
        for i in iss:
            pois1.append(pois[i])
        for i in d0:
            table0.append(pois1[i])
        print("0:", random.sample(table0, 10))
        for i in d1:
            table1.append(pois1[i])
        print("1:", random.sample(table1, 10))
        for i in d2:
            table2.append(pois1[i])
        print("2:", random.sample(table2, 10))
        for i in d3:
            table3.append(pois1[i])
        print("3:", random.sample(table3, 10))
        for i in d4:
            table4.append(pois1[i])
        print("4:", random.sample(table4, 10))

# df = pd.read_pickle("../output/encodings.pickle")
# def truncate(em):
#     return float('%.7f'%em)
# def trunc_list(ems):
#     return list(map(truncate, ems)) 
# s = [0.0] * 1536
# def to_None(ems):
#     if ems == s:
#         ems = None
#     return ems
# df["em_comp_nonfood"] = df["em_comp_nonfood"].apply(eval).apply(trunc_list).apply(to_None)
# df["em_comp_food"] = df["em_comp_food"].apply(eval).apply(trunc_list).apply(to_None)
# df["em_human_nonfood"] = df["em_human_nonfood"].apply(eval).apply(trunc_list).apply(to_None)
# df["em_human_food"] = df["em_human_food"].apply(eval).apply(trunc_list).apply(to_None)
# df["em_comp_nonfood_tags"] = df["em_comp_nonfood_tags"].apply(eval).apply(trunc_list).apply(to_None)
# df["em_comp_food_tags"] = df["em_comp_food_tags"].apply(eval).apply(trunc_list).apply(to_None)
# df["em_human_nonfood_tags"] = df["em_human_nonfood_tags"].apply(eval).apply(trunc_list).apply(to_None)
# df["em_human_food_tags"] = df["em_human_food_tags"].apply(eval).apply(trunc_list).apply(to_None)
# df.to_pickle("../output/Updated_Encodings.pickle")