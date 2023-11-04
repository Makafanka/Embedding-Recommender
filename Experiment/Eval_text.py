import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
from util import get_tags
from visualization import get_visualization
from knn import get_kmeans
# from gmm import get_gmm

ds = pd.read_pickle("sample.pickle")
pois = ds["name"]
# datafile_path_comp = "Desc_comp.csv"
# df_comp = pd.read_csv(datafile_path_comp)
# embeddings_comp = df_comp.iloc[:, 0]
datafile_path_human = "Desc_human.csv"
df_human = pd.read_csv(datafile_path_human)
embeddings_human = df_human.iloc[:, 0]

# desc = pd.read_pickle("descriptions.pickle")
# pois = desc["poi_tile"]
# datafile_path1 = "Desc1GPT.csv"
# df1 = pd.read_csv(datafile_path1)
# embeddings1 = df1.iloc[:, 0]
# datafile_path2 = "Desc2GPT.csv"
# df2 = pd.read_csv(datafile_path2)
# embeddings2 = df2.iloc[:, 0]
# datafile_path3 = "Desc3GPT.csv"
# df3 = pd.read_csv(datafile_path3)
# embeddings3 = df3.iloc[:, 0]

# datafile_p = "BaseGPT.csv"
# dff = pd.read_csv(datafile_p)
# embeddings = dff.iloc[:, 0]

# datafile_path1A = "Desc1ALBERT.csv"
# df1A = pd.read_csv(datafile_path1A)
# pip1 = df1A.iloc[:, 0]
# pipeline_res1 = []
# for em in pip1:
#     pipeline_res1.append(eval(em)[0][0])
# datafile_path2A = "Desc2ALBERT.csv"
# df2A = pd.read_csv(datafile_path2A)
# pip2 = df2A.iloc[:, 0]
# pipeline_res2 = []
# for em in pip2:
#     pipeline_res2.append(eval(em)[0][0])
# datafile_path3A = "Desc3ALBERT.csv"
# df3A = pd.read_csv(datafile_path3A)
# pip3 =  df3A.iloc[:, 0]
# pipeline_res3 = []
# for em in pip3:
#     pipeline_res3.append(eval(em)[0][0])

# datafile_p = "BaseALBERT.csv"
# dff = pd.read_csv(datafile_p)
# pips = dff.iloc[:, 0]
# pipeline_res = []
# for em in pips:
#     pipeline_res.append(eval(em)[0][0])

datafile_path = "output.csv"
df = pd.read_csv(datafile_path)
titles = df["Title"]
Ts = df["Tag"]
tags_dict = {}
tickers = []
pois = list(pois)
for i, title in enumerate(titles):
    if title in pois:
        tags_dict[title] = Ts.iloc[i]
for poi in pois:
    tickers.append(tags_dict[poi])

# dataset2 = pd.read_excel("dataset2.xlsx")
# dataset2 = dataset2.dropna(subset=["description"])
# tickers = dataset2["tag"]

# titles = dataset2["poi_tile"]
# Ts = dataset2["tag"]
# tags_dict = {}
# tickers = []
# pois = list(pois)
# for i, title in enumerate(titles):
#     if title in pois:
#         tags_dict[title] = Ts.iloc[i]
# for poi in pois:
#     tickers.append(tags_dict[poi])


# table = df1.copy()
# table.rename(columns = {'0':'Description 1'}, inplace = True)
# table['Description 2'] = embeddings2
# table['Description 3'] = embeddings3
# table['Tags'] = tickers
# table.to_pickle("tags&embeddings.pickle")  

# categories = []
# tags = []
# for ticker1 in tickers:
#     if ticker1.startswith('[{'):
#         tag = re.search(r'(?<=\{"tag":").*?(?="\})', ticker1)
#         tags.append(tag.group(0))
#         ticker = [x for x in ticker1.split(',')]
#         for s in ticker:
#             m = re.findall(r'\{"tag":"(.*)"\}', s)
#             categories.append(m)
#     else:
#         ticker = [x for x in ticker1.split(';')]
#         if ticker == []:
#             tags.append([])
#         else:
#             tags.append(ticker[0])
#         for s in ticker:
#             if s != []:
#                 s = [s]
#             categories.append(s)
# categories2 = pd.Series(categories)
# print("Categories of tags:", categories2.value_counts().head(11))

# matrix_comp = np.array(embeddings_comp.apply(eval).to_list())
# em_comp, cat_comp, tags_comp = get_tags(matrix_comp, tickers)
# # get_visualization(tickers, em_comp)
# cluster_comp, is_comp = get_kmeans(cat_comp, tags_comp, matrix_comp)
# d0 = cluster_comp[cluster_comp.cluster == 0]["data_index"]
# d1 = cluster_comp[cluster_comp.cluster == 1]["data_index"]
# d2 = cluster_comp[cluster_comp.cluster == 2]["data_index"]
# d3 = cluster_comp[cluster_comp.cluster == 3]["data_index"]
# d4 = cluster_comp[cluster_comp.cluster == 4]["data_index"]
# table0 = []
# table1 = []
# table2 = []
# table3 = []
# table4 = []
# pois1 = []
# for i in is_comp:
#     pois1.append(pois[i])
# print(is_comp)
# for i in d0:
#     table0.append(pois1[i])
# print("0:", table0)
# for i in d1:
#     table1.append(pois1[i])
# print("1:", table1)
# for i in d2:
#     table2.append(pois1[i])
# print("2:", table2)
# for i in d3:
#     table3.append(pois1[i])
# print("3:", table3)
# for i in d4:
#     table4.append(pois1[i])
# print("4:", table4)

matrix_human = np.array(embeddings_human.apply(eval).to_list())
em_human, cat_human, tags_human = get_tags(matrix_human, tickers)
# get_visualization(tickers, em_human)
cluster_human, is_human = get_kmeans(cat_human, tags_human, matrix_human)
d0 = cluster_human[cluster_human.cluster == 0]["data_index"]
d1 = cluster_human[cluster_human.cluster == 1]["data_index"]
d2 = cluster_human[cluster_human.cluster == 2]["data_index"]
d3 = cluster_human[cluster_human.cluster == 3]["data_index"]
d4 = cluster_human[cluster_human.cluster == 4]["data_index"]
table0 = []
table1 = []
table2 = []
table3 = []
table4 = []
pois1 = []
for i in is_human:
    pois1.append(pois[i])
print(is_human)
for i in d0:
    table0.append(pois1[i])
print("0:", table0)
for i in d1:
    table1.append(pois1[i])
print("1:", table1)
for i in d2:
    table2.append(pois1[i])
print("2:", table2)
for i in d3:
    table3.append(pois1[i])
print("3:", table3)
for i in d4:
    table4.append(pois1[i])
print("4:", table4)

# matrix1_GPT= np.array(embeddings1.apply(eval).to_list())
# em1_GPT, cat1_GPT, tags1_GPT = get_tags(matrix1_GPT, tickers)
# get_visualization(tickers, em1_GPT)
# cluster1_GPT, is1 = get_kmeans(cat1_GPT, tags1_GPT, matrix1_GPT)
# d0 = cluster1_GPT[cluster1_GPT.cluster == 0]["data_index"]
# d1 = cluster1_GPT[cluster1_GPT.cluster == 1]["data_index"]
# d2 = cluster1_GPT[cluster1_GPT.cluster == 2]["data_index"]
# d3 = cluster1_GPT[cluster1_GPT.cluster == 3]["data_index"]
# d4 = cluster1_GPT[cluster1_GPT.cluster == 4]["data_index"]
# table0 = []
# table1 = []
# table2 = []
# table3 = []
# table4 = []
# pois1 = []
# for i in is1:
#     pois1.append(pois[i])
# print(is1)
# for i in d0:
#     table0.append(pois1[i])
# print("0:", table0)
# for i in d1:
#     table1.append(pois1[i])
# print("1:", table1)
# for i in d2:
#     table2.append(pois1[i])
# print("2:", table2)
# for i in d3:
#     table3.append(pois1[i])
# print("3:", table3)
# for i in d4:
#     table4.append(pois1[i])
# print("4:", table4)

# t = pd.DataFrame(pd.Series(table0))
# t["1"] = pd.Series(table1)
# t["2"] = pd.Series(table2)
# t["3"] = pd.Series(table3)
# t["4"] = pd.Series(table4)
# t.to_csv('Clustering.csv', index=False)
# cluster1_GPT, is1 = get_gmm(cat1_GPT, tags1_GPT, matrix1_GPT)
# d0 = cluster1_GPT[cluster1_GPT.cluster == 0]["data_index"]
# d1 = cluster1_GPT[cluster1_GPT.cluster == 1]["data_index"]
# d2 = cluster1_GPT[cluster1_GPT.cluster == 2]["data_index"]
# d3 = cluster1_GPT[cluster1_GPT.cluster == 3]["data_index"]
# d4 = cluster1_GPT[cluster1_GPT.cluster == 4]["data_index"]
# table0 = []
# table1 = []
# table2 = []
# table3 = []
# table4 = []
# pois1 = []
# for i in is1:
#     pois1.append(pois[i])
# for i in d0:
#     table0.append(pois1[i])
# print("0:", table0)
# for i in d1:
#     table1.append(pois1[i])
# print("1:", table1)
# for i in d2:
#     table2.append(pois1[i])
# print("2:", table2)
# for i in d3:
#     table3.append(pois1[i])
# print("3:", table3)
# for i in d4:
#     table4.append(pois1[i])
# print("4:", table4)

# matrix2_GPT= np.array(embeddings2.apply(eval).to_list())
# em2_GPT, cat2_GPT, tags2_GPT = get_tags(matrix2_GPT, tickers)
# get_visualization(tickers, em2_GPT)
# get_kmeans(cat2_GPT, tags2_GPT, matrix2_GPT)
# get_gmm(cat2_GPT, tags2_GPT, matrix2_GPT)

# matrix3_GPT= np.array(embeddings3.apply(eval).to_list())
# em3_GPT, cat3_GPT, tags3_GPT = get_tags(matrix3_GPT, tickers)
# get_visualization(tickers, em3_GPT)
# cluster3_GPT, is3 = get_kmeans(cat3_GPT, tags3_GPT, matrix3_GPT)
# d0 = cluster3_GPT[cluster3_GPT.cluster == 0]["data_index"]
# d1 = cluster3_GPT[cluster3_GPT.cluster == 1]["data_index"]
# d2 = cluster3_GPT[cluster3_GPT.cluster == 2]["data_index"]
# d3 = cluster3_GPT[cluster3_GPT.cluster == 3]["data_index"]
# d4 = cluster3_GPT[cluster3_GPT.cluster == 4]["data_index"]
# table0 = []
# table1 = []
# table2 = []
# table3 = []
# table4 = []
# pois3 = []
# for i in is3:
#     pois3.append(pois[i])
# for i in d0:
#     table0.append(pois3[i])
# print("0:", table0)
# for i in d1:
#     table1.append(pois3[i])
# print("1:", table1)
# for i in d2:
#     table2.append(pois3[i])
# print("2:", table2)
# for i in d3:
#     table3.append(pois3[i])
# print("3:", table3)
# for i in d4:
#     table4.append(pois3[i])
# print("4:", table4)
# cluster3_GPT, is3 =  get_gmm(cat3_GPT, tags3_GPT, matrix3_GPT)
# d0 = cluster3_GPT[cluster3_GPT.cluster == 0]["data_index"]
# d1 = cluster3_GPT[cluster3_GPT.cluster == 1]["data_index"]
# d2 = cluster3_GPT[cluster3_GPT.cluster == 2]["data_index"]
# d3 = cluster3_GPT[cluster3_GPT.cluster == 3]["data_index"]
# d4 = cluster3_GPT[cluster3_GPT.cluster == 4]["data_index"]
# d5 = cluster3_GPT[cluster3_GPT.cluster == 5]["data_index"]
# d6 = cluster3_GPT[cluster3_GPT.cluster == 6]["data_index"]
# d7 = cluster3_GPT[cluster3_GPT.cluster == 7]["data_index"]
# d8 = cluster3_GPT[cluster3_GPT.cluster == 8]["data_index"]
# d9 = cluster3_GPT[cluster3_GPT.cluster == 9]["data_index"]
# table0 = []
# table1 = []
# table2 = []
# table3 = []
# table4 = []
# table5 = []
# table6 = []
# table7 = []
# table8 = []
# table9 = []
# pois3 = []
# for i in is3:
#     pois3.append(pois[i])
# for i in d0:
#     table0.append(pois3[i])
# print("0:", table0)
# for i in d1:
#     table1.append(pois3[i])
# print("1:", table1)
# for i in d2:
#     table2.append(pois3[i])
# print("2:", table2)
# for i in d3:
#     table3.append(pois3[i])
# print("3:", table3)
# for i in d4:
#     table4.append(pois3[i])
# print("4:", table4)
# for i in d5:
#     table5.append(pois[i])
# print("5:", table5)
# for i in d6:
#     table6.append(pois[i])
# print("6:", table6)
# for i in d7:
#     table7.append(pois[i])
# print("7:", table7)
# for i in d8:
#     table8.append(pois[i])
# print("8:", table8)
# for i in d9:
#     table9.append(pois[i])
# print("9:", table9)

# matrix_GPT= np.array(embeddings.apply(eval).to_list())
# em_GPT, cat_GPT, tags_GPT = get_tags(matrix_GPT, tickers)
# get_visualization(tickers, em_GPT)
# get_kmeans(cat_GPT, tags_GPT, matrix_GPT)
# get_gmm(cat_GPT, tags_GPT, matrix_GPT)

# matrix1_ALBERT = np.array(pipeline_res1)
# em1_AL, cat1_AL, tags1_AL = get_tags(matrix1_ALBERT, tickers)
# get_visualization(tickers, em1_AL)
# cluster1_GPT = get_kmeans(cat1_AL, tags1_AL, matrix1_ALBERT)
# d0 = cluster1_GPT[cluster1_GPT.cluster == 0]["data_index"]
# d1 = cluster1_GPT[cluster1_GPT.cluster == 1]["data_index"]
# d2 = cluster1_GPT[cluster1_GPT.cluster == 2]["data_index"]
# d3 = cluster1_GPT[cluster1_GPT.cluster == 3]["data_index"]
# d4 = cluster1_GPT[cluster1_GPT.cluster == 4]["data_index"]
# table0 = []
# table1 = []
# table2 = []
# table3 = []
# table4 = []
# for i in d0:
#     table0.append(pois[i])
# print(table0)
# for i in d1:
#     table1.append(pois[i])
# print(table1)
# for i in d2:
#     table2.append(pois[i])
# print(table2)
# for i in d3:
#     table3.append(pois[i])
# print(table3)
# for i in d4:
#     table4.append(pois[i])
# print(table4)
# cluster1_GPT = get_gmm(cat1_AL, tags1_AL, matrix1_ALBERT)
# d0 = cluster1_GPT[cluster1_GPT.cluster == 0]["data_index"]
# d1 = cluster1_GPT[cluster1_GPT.cluster == 1]["data_index"]
# d2 = cluster1_GPT[cluster1_GPT.cluster == 2]["data_index"]
# d3 = cluster1_GPT[cluster1_GPT.cluster == 3]["data_index"]
# d4 = cluster1_GPT[cluster1_GPT.cluster == 4]["data_index"]
# table0 = []
# table1 = []
# table2 = []
# table3 = []
# table4 = []
# for i in d0:
#     table0.append(pois[i])
# print(table0)
# for i in d1:
#     table1.append(pois[i])
# print(table1)
# for i in d2:
#     table2.append(pois[i])
# print(table2)
# for i in d3:
#     table3.append(pois[i])
# print(table3)
# for i in d4:
#     table4.append(pois[i])
# print(table4)
# matrix2_ALBERT = np.array(pipeline_res2)
# em2_AL, cat2_AL, tags2_AL = get_tags(matrix2_ALBERT, tickers)
# get_visualization(tickers, em2_AL)
# get_kmeans(cat2_AL, tags2_AL, matrix2_ALBERT)
# get_gmm(cat2_AL, tags2_AL, matrix2_ALBERT)
# matrix3_ALBERT = np.array(pipeline_res3)
# em3_AL, cat3_AL, tags3_AL = get_tags(matrix3_ALBERT, tickers)
# get_visualization(tickers, em3_AL)
# cluster3_GPT = get_kmeans(cat3_AL, tags3_AL, matrix3_ALBERT)
# d0 = cluster3_GPT[cluster3_GPT.cluster == 0]["data_index"]
# d1 = cluster3_GPT[cluster3_GPT.cluster == 1]["data_index"]
# d2 = cluster3_GPT[cluster3_GPT.cluster == 2]["data_index"]
# d3 = cluster3_GPT[cluster3_GPT.cluster == 3]["data_index"]
# d4 = cluster3_GPT[cluster3_GPT.cluster == 4]["data_index"]
# table0 = []
# table1 = []
# table2 = []
# table3 = []
# table4 = []
# for i in d0:
#     table0.append(pois[i])
# print(table0)
# for i in d1:
#     table1.append(pois[i])
# print(table1)
# for i in d2:
#     table2.append(pois[i])
# print(table2)
# for i in d3:
#     table3.append(pois[i])
# print(table3)
# for i in d4:
#     table4.append(pois[i])
# print(table4)
# cluster3_GPT = get_gmm(cat3_AL, tags3_AL, matrix3_ALBERT)
# d0 = cluster3_GPT[cluster3_GPT.cluster == 0]["data_index"]
# d1 = cluster3_GPT[cluster3_GPT.cluster == 1]["data_index"]
# d2 = cluster3_GPT[cluster3_GPT.cluster == 2]["data_index"]
# d3 = cluster3_GPT[cluster3_GPT.cluster == 3]["data_index"]
# d4 = cluster3_GPT[cluster3_GPT.cluster == 4]["data_index"]
# table0 = []
# table1 = []
# table2 = []
# table3 = []
# table4 = []
# for i in d0:
#     table0.append(pois[i])
# print(table0)
# for i in d1:
#     table1.append(pois[i])
# print(table1)
# for i in d2:
#     table2.append(pois[i])
# print(table2)
# for i in d3:
#     table3.append(pois[i])
# print(table3)
# for i in d4:
#     table4.append(pois[i])
# print(table4)

# matrix_ALBERT = np.array(pipeline_res)
# em_AL, cat_AL, tags_AL = get_tags(matrix_ALBERT, tickers)
# get_visualization(tickers, em_AL)
# get_kmeans(cat_AL, tags_AL, matrix_ALBERT)
# get_gmm(cat_AL, tags_AL, matrix_ALBERT)

# tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
# vis_dims1 = tsne.fit_transform(matrix_ALBERT)
# get_visualization(tickers, vis_dims1)
