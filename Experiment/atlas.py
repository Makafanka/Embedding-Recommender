from sklearn.manifold import TSNE
from nomic import atlas
import pandas as pd
import numpy as np

# num_embeddings = 1000
# embeddings = np.random.rand(num_embeddings, 256)
datafile_path = "output.csv"
df = pd.read_csv(datafile_path)
embeddings1 = df["GPT-3 Embeddings"]
embeddings2 = df["BERT Embeddings"]
matrix1 = np.array(embeddings1.apply(eval).to_list())
matrix2 = np.array(embeddings2.apply(eval).to_list())
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
vis_dims1 = tsne.fit_transform(matrix1)
vis_dims2 = tsne.fit_transform(matrix2)
# print(df["GPT-3 Embeddings"])
embedding1 = np.array(vis_dims1)
embedding2 = np.array(vis_dims2)
project1 = atlas.map_embeddings(
    embeddings=embedding1
)
project2 = atlas.map_embeddings(
    embeddings=embedding2
)
