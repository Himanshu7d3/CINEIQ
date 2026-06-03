import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFiltering:

    def __init__(self, df):
        self.df = df.copy()
        # self.df["tf_idf_title"] = self.df["title"].apply(self._normalize)

        # build or load TF-IDF once
        self.tfidf_matrix, self.title_to_idx = self._build_or_load_tfidf()

    # general function to normalize text (lowercase, strip, remove extra spaces)
    def _normalize(self, text):
        text = str(text).lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    # build or load TF-IDF from pickle file
    def _build_or_load_tfidf(self):

        pickle_path = "tfidf_data.pkl"

        # load from pickle if exists and shape matches
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)

            if data.get("shape") == self.df.shape:
                return data["matrix"], data["title_to_idx"]

        # build tfidf_data.pkl if not exists or shape mismatch
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.df["tags"])

        title_to_idx = pd.Series(
            self.df.index,
            index=self.df["tf_idf_title"]
        ).to_dict()

        # save to pickle for future use
        data = {
            "matrix": tfidf_matrix,
            "title_to_idx": title_to_idx,
            "shape": self.df.shape
        }

        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)

        return tfidf_matrix, title_to_idx

    # Recommendation by TF-IDF
    def recommend_tfidf(self, movie_title, top_n=10):

        movie_title = self._normalize(movie_title)

        if movie_title not in self.title_to_idx:
            return pd.DataFrame(columns=["title", "score"])

        idx = self.title_to_idx[movie_title]

        sim = cosine_similarity(
            self.tfidf_matrix,
            self.tfidf_matrix[idx]
        ).ravel()

        sim[idx] = -1

        top_idx = np.argsort(sim)[::-1][:top_n]

        result = self.df.iloc[top_idx][["title"]].copy()
        result["score"] = sim[top_idx]

        return result.reset_index(drop=True)

    # tfidf scores for candidates (for hybrid)
    def tfidf_scores_for_candidates(self, source_movie, candidate_movies=None, top_n=100):

        source_movie = self._normalize(source_movie)

        if source_movie not in self.title_to_idx:
            return pd.DataFrame(columns=["movie", "content_score"])

        source_idx = self.title_to_idx[source_movie]

        target_genres = set(str(self.df.loc[source_idx, "genres"]).lower().split())
        target_actors = set(str(self.df.loc[source_idx, "Actors"]).lower().split())
        target_director = str(self.df.loc[source_idx, "Director"]).lower()

        if candidate_movies is None:
            candidate_movies = self.df["title"].tolist()

        results = []

        for movie in candidate_movies:

            movie_norm = self._normalize(movie)
            idx = self.title_to_idx.get(movie_norm)

            if idx is None:
                continue

            tfidf_score = cosine_similarity(
                self.tfidf_matrix[source_idx],
                self.tfidf_matrix[idx]
            )[0][0]

            genres = set(str(self.df.loc[idx, "genres"]).lower().split())
            actors = set(str(self.df.loc[idx, "Actors"]).lower().split())
            director = str(self.df.loc[idx, "Director"]).lower()

            genre_score = len(target_genres & genres) / max(len(target_genres | genres), 1)
            actor_score = len(target_actors & actors) / max(len(target_actors | actors), 1)
            director_score = 1.0 if director == target_director else 0.0

            final_score = (
                0.65 * tfidf_score +
                0.20 * genre_score +
                0.10 * actor_score +
                0.05 * director_score
            )

            results.append({
                "movie": movie,
                "content_score": final_score
            })

        return pd.DataFrame(results).sort_values(
            "content_score",
            ascending=False
        ).head(top_n).reset_index(drop=True)