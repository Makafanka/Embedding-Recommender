from key1 import API_KEY1
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
import numpy as np
import pandas as pd
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

openai.api_key = API_KEY1

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))

def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    """
    Generate one list of embedding for one description
    :param text: str, one piece of text or description that needs to be embedded
    :param model: str ("text-embedding-ada-002"), the name of the GPT-3 model we are using
    :return: list[float], a list of embeddings
    """
    # if text == "NONE" or text == "":
    #     return [0]
    # else:
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]

def get_reduced_em(matrix):
    # Create a t-SNE model and transform the data
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)
    embeddings = np.array(vis_dims)
    return embeddings


def get_tags(matrix, tickers1):
    """
    Use tSNE to reduce the dimension of embeddings, and select the main tag among all tags for each description
    :param matrix: np.array, value of embeddings for all descriptions
    :param tickers1: list[str], list of all tags for each of the descriptions
    :return embeddings: np.array, embeddings reduced into 2-D, coordinates
    :return categories_ar: np.array, all possible tags
    :return tags: list[str], list of one picked tag for each of the descriptions
    """
    # Create a t-SNE model and transform the data
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(matrix)
    embeddings = np.array(vis_dims)

    # Extract main tags from original tag lists
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
        elif ticker1.startswith('['):
            ticker1 = ticker1.strip("[]")
            ticker = [x.strip("\'") for x in ticker1.split(', ')]
            if ticker == []:
                tags.append([])
            else:
                tags.append(ticker[0])
            for s in ticker:
                if s != []:
                    s = [s]
                categories.append(s)
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
    categories2 = pd.Series(categories)
    
    return embeddings, categories_ar, categories2, tags
    # return categories_ar, tags

def get_visualization(tickers1, vis_dims1, tag1, tag2, tag3):
    """
    Generate 2-D visualization for two similar tags and two different tags
    :param tickers1: list[str], list of all tags for each of the descriptions
    :param vis_dims1: np.array, embeddings reduced into 2-D, coordinates
    :param tag1: str, the first tag
    :param tag2: str, a tag that is similar to the first tag
    :param tag1: str, a tag that is different from the first tag
    """
    colors = ["darkorange", "turquoise"]
    x11 = []
    y11 = []
    x10 = []
    y10 = []
    x21 = []
    y21 = []
    x20 = []
    y20 = []

    for i, tag in enumerate(tickers1):
        if (tag1 in tag) or (tag2 in tag):
            x, y = vis_dims1[i] 
            if (tag1 in tag) and not (tag2 in tag):
                x10.append(x)
                y10.append(y)
            elif (tag2 in tag) and not (tag1 in tag): 
                x11.append(x)
                y11.append(y)
            else:
                x10.append(x)
                y10.append(y)
                x11.append(x)
                y11.append(y)

    for i, tag in enumerate(tickers1):
        if (tag1 in tag) or (tag3 in tag):
            x, y = vis_dims1[i] 
            if (tag1 in tag) and not (tag3 in tag):
                x20.append(x)
                y20.append(y)
            elif (tag3 in tag) and not (tag1 in tag): 
                x21.append(x)
                y21.append(y)
            else:
                x20.append(x)
                y20.append(y)
                x21.append(x)
                y21.append(y)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for color in colors:
        if color == "darkorange":
            ax1.scatter(x10, y10, color=color, alpha=0.3)
        else:
            ax1.scatter(x11, y11, color=color, alpha=0.3)
    ax1.legend([tag1, tag2], loc="upper right")
    plt.title("Visualization of " + tag1 + " VS " + tag2)
    # plt.savefig('output/plot1.png')
    plt.savefig('../output/plot1.png')

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for color in colors:
        if color == "darkorange":
            ax2.scatter(x20, y20, color=color, alpha=0.3)
        else:
            ax2.scatter(x21, y21, color=color, alpha=0.3)
    ax2.legend([tag1, tag3], loc ="upper right")
    plt.title("Visualization of " + tag1 + " VS " + tag3)
    # plt.savefig('output/plot2.png')
    plt.savefig('../output/plot2.png')

    plt.close()
    # plt.show()
    

def get_kmeans(categories_ar, tags, matrix):
    """
    Do k-means with 5 clusters, get the silhouette score, and get the cluster map
    :param categories_ar: np.array, all possible tags
    :param tags: list[str], list of one picked tag for each of the descriptions
    :param matrix: np.array, value of embeddings for all descriptions
    :return cluster_map: pd.dataframe, a map of indices of tags to each cluster
    :return indices: list[int], the list of indices for each tag
    """
    categories = categories_ar
    values, counts = np.unique(categories, return_counts=True)
    ind = np.argpartition(-counts, kth=10)[:10]
    new_cat = values[ind]
    embed = []
    indices = []
    for i, tag in enumerate(tags):
        if tag in new_cat:
            embed.append(matrix[i])
            indices.append(i)
    embed = np.array(embed)
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(embed)
    embeddings = pd.DataFrame(vis_dims)

    # do k-means for n=5 and get the clusters
    X = embeddings
    KMean5= KMeans(n_clusters=5)
    km = KMean5.fit(X)
    label5=KMean5.predict(X)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = X.index.values
    cluster_map['cluster'] = km.labels_
    print("Silhouette Score with n=5", silhouette_score(X, label5))
    return cluster_map, indices