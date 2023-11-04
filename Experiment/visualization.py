import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import re

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

# Load the embeddings
datafile_path = "output.csv"
df = pd.read_csv(datafile_path)
# embedding1 = df["GPT-3 Embeddings"]
# em2 = df["BERT Embeddings"]
# embedding1 = [] 
# # print(eval(em2[0])[0][0])
# for em in em2:
#     embedding1.append(eval(em)[0][0])


# # Convert to a list of lists of floats
# matrix1 = np.array(embedding1.apply(eval).to_list())
# matrix1 = np.array(embedding1)


# Create a t-SNE model and transform the data
# tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
# vis_dims1 = tsne.fit_transform(matrix1)
# df["embed_vis1"] = vis_dims1.tolist()
# vis_dims2 = tsne.fit_transform(matrix2)
# # df["embed_vis2"] = vis_dims2.tolist()
# # print(vis_dims.shape)

# tickers1 = df["Tag"]
# categories = []
# tags = []
# for ticker1 in tickers1:
#     if ticker1.startswith('[{'):
#         tag = re.search(r'(?<=\{"tag":").*?(?="\})', ticker1)
#         tags.append(tag.group(0))
#         ticker = [x for x in ticker1.split(',')]
#         for s in ticker:
#             # m = re.search(r'(?<=\{"tag":")\S+', s)
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
# categories1 = np.array(categories)
# categories1 = np.unique(categories1)



def get_visualization(tickers1, vis_dims1):
    # cmap = plt.get_cmap('hsv', len(categories1))
    colors = ["darkorange", "turquoise"]
    # x1 = [x for x,y in vis_dims1]
    # y1 = [y for x,y in vis_dims1]
    x11 = []
    y11 = []
    x10 = []
    y10 = []
    x21 = []
    y21 = []
    x20 = []
    y20 = []
    # num_tags = []

    for i, tag in enumerate(tickers1):
        if ("Cocktail Lounges" in tag) or ("Bars" in tag):
            x, y = vis_dims1[i] 
            if ("Cocktail Lounges" in tag) and not ("Bars" in tag):
                # num_tags.append(0)
                x10.append(x)
                y10.append(y)
            elif ("Bars" in tag) and not ("Cocktail Lounges" in tag): 
                # num_tags.append(1)
                x11.append(x)
                y11.append(y)
            else:
                x10.append(x)
                y10.append(y)
                x11.append(x)
                y11.append(y)
                # num_tags.append(0)
                # num_tags.append(1)

    for i, tag in enumerate(tickers1):
        if ("Cocktail Lounges" in tag) or ("Mixed Clothing" in tag):
            x, y = vis_dims1[i] 
            if ("Cocktail Lounges" in tag) and not ("Mixed Clothing" in tag):
                x20.append(x)
                y20.append(y)
            elif ("Mixed Clothing" in tag) and not ("Cocktail Lounges" in tag): 
                x21.append(x)
                y21.append(y)
            else:
                x20.append(x)
                y20.append(y)
                x21.append(x)
                y21.append(y)

    # cmap = matplotlib.colors.ListedColormap(colors)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    # ax.scatter(x1, y1, c=num_tags, cmap=cmap, alpha=0.3)
    # for i, n in enumerate(num_tags):
    #     ax.scatter(x1[i], y1[i], color=colors[n], alpha=0.3)

    for color in colors:
        if color == "darkorange":
            ax1.scatter(x10, y10, color=color, alpha=0.3)
        else:
            ax1.scatter(x11, y11, color=color, alpha=0.3)

            
    # Plot each sample category individually such that we can set label name.
    # for score in [0, 1]:
    # for i, cat in enumerate(categories1):
    #     avg_x = np.array(x1)[num_tags==i].mean()
    #     avg_y = np.array(y1)[num_tags==i].mean()
    #     color = cmap(i)
    #     # color = colors[score]
    #     plt.scatter(avg_x, avg_y, marker='x', color=color, s=100)

        # sub_matrix = np.array(df[cat in df["Tag"]]["embed_vis"].to_list())
        # x = sub_matrix[:, 0]
        # y = sub_matrix[:, 1]
        # colors = [cmap(i/len(categories))] * len(sub_matrix)
        # plt.scatter(x, y, color=colors, label=cat)
    ax1.legend(["Cocktail Lounges", "Bars"], loc="upper right")
    plt.title("Visualization of Cocktail Lounges VS Bars")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for color in colors:
        if color == "darkorange":
            ax2.scatter(x20, y20, color=color, alpha=0.3)
        else:
            ax2.scatter(x21, y21, color=color, alpha=0.3)
    ax2.legend(["Cocktail Lounges", "Mixed Clothing"], loc ="upper right")
    plt.title("Visualization of Cocktail Lounges VS Mixed Clothing")
    plt.show()

    # x2 = [x for x,y in vis_dims2]
    # y2 = [y for x,y in vis_dims2]

    # plt.scatter(x2, y2, c=num_tags, cmap=cmap, alpha=0.3)
    # # # Plot each sample category individually such that we can set label name.
    # # for i, cat in enumerate(categories):
    # #     avg_x = np.array(x2)[num_tags==i].mean()
    # #     avg_y = np.array(y2)[num_tags==i].mean()
    # #     color = cmap(i)
    # #     plt.scatter(avg_x, avg_y, marker='x', color=color, s=100)
    # plt.show()

# get_visualization(tickers1=tickers1, vis_dims1=vis_dims1)