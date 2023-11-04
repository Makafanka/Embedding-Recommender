import pandas as pd
import numpy as np
from key1 import API_KEY1
import openai
from util import get_embedding
from util import get_extractor

openai.api_key = API_KEY1

ds = pd.read_pickle("sample.pickle")
ds.loc[264, "descr_human_nonfood"] = "Placeholder"
data_comp = ds["descr_comp_nonfood"]
data_human = ds["descr_human_nonfood"]
embeddings_comp = []
embeddings_human = []

# dataset2 = pd.read_excel("dataset2.xlsx")
# dataset2 = dataset2.dropna(subset=["description"])
# desc2 = dataset2["description"]
# desc2 = desc2.to_json(orient='records')
# oth2 = dataset2[["poi_tile", "tag", "description"]]
# oth2.columns = ["Title", "Tag", "Description"]

# desc = pd.read_pickle("descriptions.pickle")
# poi = desc["poi_tile"]
# data1 = desc["temp0.50_topp0.80_expln"]
# data2 = desc["temp0.20_topp0.60_expln"]
# data3 = desc["temp0.20_topp0.80_expln"]

# embeddings1 = []
# pipeline_res1 = []
# embeddings2 = []
# pipeline_res2 = []
# embeddings3 = []
# pipeline_res3 = []
# embeddings = []
# pipeline_res = []

# extractor = get_extractor()

# ds = data3.iloc[264]
# ds = "placeholder"
# print(ds)
# embeddings3.append(get_embedding(ds, model="text-embedding-ada-002"))
# print(embeddings3)
# pipeline_res1.append(extractor(ds, truncation=True))
# print(pipeline_res1)
# ds = desc2.iloc[0]
# embeddings.append(get_embedding(ds, model="text-embedding-ada-002"))
# print(embeddings)

# for data in data_comp:
#     embeddings_comp.append(get_embedding(data, model="text-embedding-ada-002"))
# embeddings_comp = pd.DataFrame(pd.Series(embeddings_comp))
# embeddings_comp.to_csv('Desc_comp.csv', index=False)

for data in data_human:
    embeddings_human.append(get_embedding(data, model="text-embedding-ada-002"))
embeddings_human = pd.DataFrame(pd.Series(embeddings_human))
embeddings_human.to_csv('Desc_human.csv', index=False)

# for ds in desc2:
    # embeddings.append(get_embedding(ds, model="text-embedding-ada-002"))
#     pipeline_res.append(extractor(ds, truncation=True))
# pipeline_res = pd.DataFrame(pd.Series(pipeline_res))
# pipeline_res.to_csv('BaseALBERT.csv', index=False)
# embeddings = pd.DataFrame(pd.Series(embeddings))
# embeddings.to_csv('BaseGPT.csv', index=False)

# for ds in data1:
    # embeddings1.append(get_embedding(ds, model="text-embedding-ada-002"))
    # pipeline_res1.append(extractor(ds, truncation=True))
# embeddings1 = pd.DataFrame(pd.Series(embeddings1))
# embeddings1.to_csv('Desc1GPT.csv', index=False)
# pipeline_res1 = pd.DataFrame(pd.Series(pipeline_res1))
# pipeline_res1.to_csv('Desc1ALBERT.csv', index=False)

# for ds in data2:
#     embeddings2.append(get_embedding(ds, model="text-embedding-ada-002"))
    # pipeline_res2.append(extractor(ds, truncation=True))
# embeddings2 = pd.DataFrame(pd.Series(embeddings2))
# embeddings2.to_csv('Desc2GPT.csv', index=False)
# pipeline_res2 = pd.DataFrame(pd.Series(pipeline_res2))
# pipeline_res2.to_csv('Desc2ALBERT.csv', index=False)
# data3.update(pd.Series(["placeholder"], index=[264]))

# for ds in data3:
#     embeddings3.append(get_embedding(ds, model="text-embedding-ada-002"))
    # pipeline_res3.append(extractor(ds, truncation=True))
# embeddings3 = pd.DataFrame(pd.Series(embeddings3))
# embeddings3.to_csv('Desc3GPT.csv', index=False)
# pipeline_res3 = pd.DataFrame(pd.Series(pipeline_res3))
# pipeline_res3.to_csv('Desc3ALBERT.csv', index=False)
