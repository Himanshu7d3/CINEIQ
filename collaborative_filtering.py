import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import os
import pickle


class collaborative_filtering:

    # training the svd model
    def train_svd(self, ratings, pickle_path="svd_data.pkl"):

        # Create categorical mappings
        user_ids = ratings["userId"].astype("category")
        movie_ids = ratings["movieId"].astype("category")

        user_map = user_ids.cat.categories
        movie_map = movie_ids.cat.categories

        user_index = user_ids.cat.codes
        movie_index = movie_ids.cat.codes

        # Create sparse rating matrix
        R = csr_matrix((ratings["rating"], (user_index, movie_index)))

        # Train SVD
        svd = TruncatedSVD(n_components=80, random_state=42)
        U = svd.fit_transform(R)

        # Save everything
        data = {
            "svd": svd,
            "U": U,
            "user_map": user_map,
            "movie_map": movie_map
        }

        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)

        print("SVD model trained and saved successfully.")

    # Recommendation by svd
    def recommend_svd(self, new_df, ratings, user_id, top_n=10):

        pickle_path = "svd_data.pkl"

        # if SVD model not found, train it first
        if not os.path.exists(pickle_path):
            self.train_svd(ratings, pickle_path)

        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        svd = data["svd"]
        U = data["U"]
        user_map = data["user_map"]
        movie_map = data["movie_map"]

        # Check user exists
        if user_id not in user_map:
            return pd.DataFrame(columns=["movie", "score"])

        # User index
        user_idx = np.where(user_map == user_id)[0][0]
        user_vector = U[user_idx]

        # Predict scores
        scores = user_vector @ svd.components_

        # Top recommendations
        top_idx = np.argsort(scores)[::-1][:top_n]
        recommended_movie_ids = movie_map[top_idx]

        # MovieId → Title mapping
        movie_id_to_title = dict(zip(new_df["movieId"], new_df["title"]))

        return pd.DataFrame({
            "movie": [
                movie_id_to_title.get(mid, str(mid))
                for mid in recommended_movie_ids
            ],
            "score": scores[top_idx]
        })