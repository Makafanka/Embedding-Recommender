import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from util import get_tags
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

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

embeddings, categories_ar, tags = get_tags(matrix, tickers1)

def get_gmm(categories_ar, tags, matrix):
    categories = categories_ar
    values, counts = np.unique(categories, return_counts=True)

    ind = np.argpartition(-counts, kth=10)[:10]
    new_cat = values[ind]
    # new_cat = []
    # for c in cat:
    #     if c != "[]":
    #         new_cat.append(c)
    # print(new_cat)  # prints the 10 most frequent elements
    # print(tags)
    # new_tags = []
    embed = []
    indices = []
    for i, tag in enumerate(tags):
        if tag in new_cat:
            # new_tags.append(tag)
            embed.append(matrix[i])
            indices.append(i)
    print(len(indices))
    embed = np.array(embed)
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(embed)
    embeddings = pd.DataFrame(vis_dims)

    # d = {t: i for i, t in enumerate(new_cat)}
    # num_tags = [d[t] for t in new_tags]
    X = embeddings
    # y = num_tags

    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    # GMM10 = GaussianMixture(n_components=10, random_state=0).fit(X)
    # y_pred_10 = GMM10.predict(X)
    GMM5 = GaussianMixture(n_components=5, random_state=0).fit(X)
    y_pred_5 = GMM5.predict(X)
    # GMM2 = GaussianMixture(n_components=2, random_state=0).fit(X)
    # y_pred_2 = GMM2.predict(X)

    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = X.index.values
    cluster_map['cluster'] = y_pred_5
    # print(cluster_map[cluster_map.cluster == 0])
    # print(cluster_map[cluster_map.cluster == 1])
    # print(cluster_map[cluster_map.cluster == 2])
    # print(cluster_map[cluster_map.cluster == 3])
    # print(cluster_map[cluster_map.cluster == 4])
    return cluster_map, indices
    # print("silhouttte with n = 10: ", silhouette_score(X, y_pred_10))
    # print("AIC with n = 10: ", GMM10.aic(X))
    # print("silhouttte with n = 5: ", silhouette_score(X, y_pred_5))
    # print("AIC with n = 5: ", GMM5.aic(X))
    # print("silhouttte with n = 2: ", silhouette_score(X, y_pred_2))
    # print("AIC with n = 2: ", GMM2.aic(X))