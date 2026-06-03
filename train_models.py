import os
import pickle
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


def build_all_models(df, ratings):
    tfidf_path = "tfidf_data.pkl"
    df["tags"] = df["tags"].fillna("")

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["tags"])

    title_to_idx = pd.Series(
        df.index,
        index=df["tf_idf_title"]
    ).to_dict()

    tfidf_data = {
        "matrix": tfidf_matrix,
        "title_to_idx": title_to_idx,
        "shape": df.shape
    }

    with open(tfidf_path, "wb") as f:
        pickle.dump(tfidf_data, f)

    print("✅ TF-IDF model saved")

    svd_path = "svd_data.pkl"

    user_ids = ratings["userId"].astype("category")
    movie_ids = ratings["movieId"].astype("category")

    user_map = user_ids.cat.categories
    movie_map = movie_ids.cat.categories

    user_index = user_ids.cat.codes
    movie_index = movie_ids.cat.codes

    R = csr_matrix((ratings["rating"], (user_index, movie_index)))

    svd = TruncatedSVD(n_components=80, random_state=42)
    U = svd.fit_transform(R)

    svd_data = {
        "svd": svd,
        "U": U,
        "user_map": user_map,
        "movie_map": movie_map
    }

    with open(svd_path, "wb") as f:
        pickle.dump(svd_data, f)

    print("✅ SVD model saved")

    return "Both TF-IDF and SVD models built successfully"