import openai
import pandas as pd
import numpy as np
from transformers import pipeline
from key1 import API_KEY1
from tenacity import retry, wait_random_exponential, stop_after_attempt

openai.api_key = API_KEY1

# df = pd.read_csv("output.csv")
# print(len(df["BERT Embeddings"][1]))
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))

def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

def get_extractor(model="albert-base-v2", task="feature-extraction", tokenizer="albert-base-v2"):
    return pipeline(model=model, task=task, tokenizer=tokenizer)


def get_data():
    dataset1 = pd.read_csv("dataset1.csv")
    dataset1 = dataset1.dropna(subset=["Description"])
    desc1 = dataset1["Description"]
    # desc1 = desc1.to_json(orient='records')
    oth1 = dataset1[["Title", "Tag", "Description", "City", "Country"]]

    dataset2 = pd.read_excel("dataset2.xlsx")
    dataset2 = dataset2.dropna(subset=["description"])
    desc2 = dataset2["description"]
    # desc2 = desc2.to_json(orient='records')
    oth2 = dataset2[["poi_tile", "tag", "description"]]
    oth2.columns = ["Title", "Tag", "Description"]
    city2 = ["New York City" for i in range(len(oth2.index))]
    oth2["City"] = city2
    country2 = ["United States" for i in range(len(oth2.index))]
    oth2["Country"] = country2

    dataset3 = pd.read_csv("dataset3.csv")
    dataset3 = dataset3.dropna(subset=["Description"])
    desc3 = dataset3["Description"]
    oth3 = dataset3[["Title", "Tag", "Description", "City", "Country"]]

    descs = [desc1, desc2, desc3]
    data = pd.concat(descs)
    oths = [oth1, oth2, oth3]
    table = pd.concat(oths)
    return data, table

def main():
    embeddings = []
    pipeline_res = []
    extractor = get_extractor()
    data, table = get_data()
    for ds in data:
        embeddings.append(get_embedding(ds, model="text-embedding-ada-002"))
        pipeline_res.append(extractor(ds, truncation=True))
    table["GPT-3 Embeddings"] = embeddings
    table["BERT Embeddings"] = pipeline_res

    table.to_csv('output.csv', index=False)

if __name__ == "__main__":
    main()





