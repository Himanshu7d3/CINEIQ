import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
class collaborative_filtering:
    def recommend_svd(self,new_df,ratings,user_id,top_n=10):
        user_ids = ratings['userId'].astype('category')
        movie_ids = ratings['movieId'].astype('category')

        user_map = user_ids.cat.categories
        movie_map = movie_ids.cat.categories

        user_index = user_ids.cat.codes
        movie_index = movie_ids.cat.codes
        R = csr_matrix(
    (ratings['rating'], (user_index, movie_index))
)
        svd = TruncatedSVD(n_components=80, random_state=42)
        U = svd.fit_transform(R)
        movie_id_to_title = dict(zip(new_df.movieId, new_df.title))
        # get user index
        user_idx = np.where(user_map == user_id)[0][0]
        user_vector = U[user_idx]
        
        scores = user_vector @ svd.components_
        top_idx = np.argsort(scores)[::-1][:top_n]
        
        movie_ids = movie_map[top_idx]
        
        return pd.DataFrame({"movie":[movie_id_to_title.get(mid,mid)for mid in movie_ids],"score":[scores[idx] for idx in top_idx]})

