import numpy as np
import pandas as pd
from openai.embeddings_utils import cosine_similarity

class Recommender:
    def __init__(self):
        print("Recommender Class")

    def recommend(self, profile, stacked_profiles, topk, topp, temp, pw_func, select_len):
        """
        Generate the recommendation based on a specific profile
        :param profile: np.array, the embedding we are focusing on
        :param stacked_profiles: 2D np.array, all other embeddings
        :param topk: int, the number of pois we use
        :param topp: float, probability bound
        :param temp: float, smoothing parameter
        :param pw_func: function, a function used to apply penalty_weight to the scores
        :param select_len: int, the number of pois we want to randomly select in the last step
        :return scores: list, the final scores of the selected pois
        :return map_dict: dict, the map of indices (in the original stacked_profiles) and scores
        """

        # Calculate similarity
        similarity = np.dot(profile, np.transpose(stacked_profiles))

        # Build a map_dict
        map_dict = {i:s for i, s in enumerate(similarity)}
        

        # Select topk pois
        # res = np.sort(similarity)[::-1][:topk]
        idx = np.flip(np.argsort(list(map_dict.values())))[:topk]
        res = similarity[idx]
        
        # Get penalty weight
        res = pw_func(res) 
        map_dict = {i:s for i, s in zip(idx, res)}
        idx = np.flip(np.argsort(list(map_dict.values())))
        res = np.sort(res)
        
        rec = self.softmax_sm(res, temp) # Softmax with smoothing
        
        filtered_rec, idx = self.get_topp(rec, topp, idx) # Filter out pois after topp

        rec_score = self.softmax(filtered_rec) # Softmax
        map_dict = {i:s for i, s in zip(idx, rec_score)}

        idx = np.random.choice(len(idx), size=select_len, replace=False) # Sample from those probabilities
        scores = []
        for i in idx:
            scores.append(map_dict[i])
        map_dict = {i:s for i, s in zip(idx, rec_score)}
        
        return scores, map_dict


    def softmax_sm(self, x, t):
        """ 
        Softmax with smoothing, higher the t value the higher the smoothing
        :param x: float, original distance
        :param t: float, smoothing parameter
        :return: the smoothed softmax values that sums to 1
        """
        e_x = np.exp((x - np.max(x)) / t)
        return e_x / e_x.sum(axis=0)

    def softmax(self, x):
        """
        Compute softmax values for each sets of scores in x.
        :param x: float, original data
        :return: the softmax values that sums to 1
        """
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def get_topp(self, lst, bound, idx):
        total = 0.0
        selected_elements = []
        new_idx = []
        for i, element in enumerate(lst):
            if total + element <= bound:
                selected_elements.append(element)
                new_idx.append(idx[i])
                total += element
            else:
                break
        return selected_elements, new_idx

def strtoint(s):
    news = s[1:-1]
    return np.fromstring(news, dtype=float, sep=', ')

# df = pd.read_csv("embeddings_df.csv")
# profile = strtoint(df["embeddings"].iloc[0])
# profile_series = df["embeddings"].iloc[1:]
# stacked_profiles = []
# for x in profile_series:
#     y = strtoint(x)
#     stacked_profiles.append(y)
# stacked_profiles = np.array(stacked_profiles)
# e = Recommender()
# topk = 80
# topp = 0.8
# temp = 0.5
# def pw_func(v):
#     return v / 2
# e.recommend(profile, stacked_profiles, 80, 0.8, 0.5, pw_func, 20)

# def cos_sim(p):
#     return cosine_similarity(p, profile)
# cos_sim_np = np.vectorize(cos_sim)
# print(cos_sim(stacked_profiles[55]))