import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from util import get_tags

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

def get_kmeans(categories_ar, tags, matrix):
    categories = categories_ar
    values, counts = np.unique(categories, return_counts=True)

    ind = np.argpartition(-counts, kth=10)[:10]
    new_cat = values[ind]
    # new_cat = []
    # for c in cat:
    #     if c != "[]":
    #         new_cat.append(c)
    # print(new_cat)  # prints the 10 most frequent elements

    # new_tags = []
    embed = []
    indices = []
    # print(len(tags))
    for i, tag in enumerate(tags):
        if tag in new_cat:
            # new_tags.append(tag)
            embed.append(matrix[i])
            indices.append(i)
    embed = np.array(embed)
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims = tsne.fit_transform(embed)
    embeddings = pd.DataFrame(vis_dims)
    # d = {t: i for i, t in enumerate(new_cat)}
    # num_tags = [d[t] for t in new_tags]

    X = embeddings
    # y = num_tags

    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    # knn10 = KNeighborsClassifier(n_neighbors = 10)
    # knn5 = KNeighborsClassifier(n_neighbors = 5)
    # knn1 = KNeighborsClassifier(n_neighbors=1)

    # knn10.fit(X_train, y_train)
    # knn5.fit(X_train, y_train)
    # knn1.fit(X_train, y_train)

    # y_pred_10 = knn10.predict(X_test)
    # y_pred_5 = knn5.predict(X_test)
    # y_pred_1 = knn1.predict(X_test)

    # print("Accuracy with k=10", accuracy_score(y_test, y_pred_10)*100)
    # print("Accuracy with k=5", accuracy_score(y_test, y_pred_5)*100)
    # print("Accuracy with k=1", accuracy_score(y_test, y_pred_1)*100)

    # print("Silhouette Score with k=10", silhouette_score(X_test, y_pred_10))
    # print("Silhouette Score with k=5", silhouette_score(X_test, y_pred_5))
    # print("Silhouette Score with k=1", silhouette_score(X_test, y_pred_1))

    # KMean10= KMeans(n_clusters=10)
    # KMean10.fit(X)
    # label10=KMean10.predict(X)

    KMean5= KMeans(n_clusters=5)
    km = KMean5.fit(X)
    label5=KMean5.predict(X)

    # KMean10= KMeans(n_clusters=10)
    # km = KMean10.fit(X)

    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = X.index.values
    cluster_map['cluster'] = km.labels_
    # print(cluster_map[cluster_map.cluster == 0])
    # print(cluster_map[cluster_map.cluster == 1])
    # print(cluster_map[cluster_map.cluster == 2])
    # print(cluster_map[cluster_map.cluster == 3])
    # print(cluster_map[cluster_map.cluster == 4])
    # u_labels = np.unique(label5)
    # for i in u_labels:
    #     plt.scatter(df[label5 == i , 0] , df[label5 == i , 1] , label = i)
    # plt.legend()
    # plt.show()

    # KMean2= KMeans(n_clusters=2)
    # KMean2.fit(X)
    # label2=KMean2.predict(X)

    # print("Silhouette Score with n=10", silhouette_score(X, label10))
    print("Silhouette Score with n=5", silhouette_score(X, label5))
    # print("Silhouette Score with n=2", silhouette_score(X, label2))
    # label5 = pd.DataFrame(pd.Series(label5))
    # label5.to_csv('label.csv', index=False)
    
    return cluster_map, indices
    # plt.figure(figsize = (15,5))
    # plt.subplot(2,2,1)
    # plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_10, marker= '*', s=100,edgecolors='black')
    # plt.title("Predicted values with k=10", fontsize=20)

    # plt.subplot(2,2,2)
    # plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_5, marker= '*', s=100,edgecolors='black')
    # plt.title("Predicted values with k=5", fontsize=20)

    # plt.subplot(2,2,3)
    # plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_1, marker= '*', s=100,edgecolors='black')
    # plt.title("Predicted values with k=1", fontsize=20)
    # plt.show()

# get_kmeans(categories_ar, tags, matrix)