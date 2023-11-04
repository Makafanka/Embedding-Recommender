import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import openai
import re
from transformers import pipeline
from tenacity import retry, wait_random_exponential, stop_after_attempt

datafile_path = "output.csv"
df = pd.read_csv(datafile_path)
embedding = df["GPT-3 Embeddings"]
# em2 = df["BERT Embeddings"]
# embedding = [] 
# for em in em2:
#     embedding.append(eval(em)[0][0])

# Convert to a list of lists of floats
matrix = np.array(embedding.apply(eval).to_list())
# matrix = np.array(embedding)
tickers1 = df["Tag"]



@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))

def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

def get_extractor(model="albert-base-v2", task="feature-extraction", tokenizer="albert-base-v2"):
    return pipeline(model=model, task=task, tokenizer=tokenizer)

def get_tags(matrix, tickers1):
    # Create a t-SNE model and transform the data
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)
    embeddings = np.array(vis_dims)

    categories = []
    tags = []
    for ticker1 in tickers1:
        if ticker1.startswith('[{'):
            tag = re.search(r'(?<=\{"tag":").*?(?="\})', ticker1)
            tags.append(tag.group(0))
            ticker = [x for x in ticker1.split(',')]
            for s in ticker:
                m = re.findall(r'\{"tag":"(.*)"\}', s)
                categories.append(m)
        else:
            ticker = [x for x in ticker1.split(';')]
            if ticker == []:
                tags.append([])
            else:
                tags.append(ticker[0])
            for s in ticker:
                if s != []:
                    s = [s]
                categories.append(s)
    categories_ar = np.array(categories)
    # categories_un = np.unique(categories)
    # categories2 = pd.Series(categories)
    # print("Categories of tags:", categories2.value_counts().head(20))

    return embeddings, categories_ar, tags

# get_tags(matrix, tickers1)