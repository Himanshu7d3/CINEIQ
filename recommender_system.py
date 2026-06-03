import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

from content_based_filtering import ContentBasedFiltering
from collaborative_filtering import collaborative_filtering


class recommender_system:

    def __init__(self, df, ratings):
        self.df = df.copy()
        self.ratings = ratings

        self.cb = ContentBasedFiltering(self.df)
        self.cf = collaborative_filtering()

    def hybrid_recommendation(self, user_id=None, movie_title=None, top_n=10):

        # only user_id provided, no movie_title
        if user_id is not None and movie_title is None:

            svd_df = self.cf.recommend_svd(
                self.df,
                self.ratings,
                user_id=user_id,
                top_n=10
            )
            svd_df = svd_df[svd_df["movie"].isin(self.df["title"])]

            return svd_df.rename(columns={
                "movie": "title",
                "score": "svd_score"
            }).head(top_n).reset_index(drop=True)

        # only movie_title provided, no user_id
        if movie_title is not None and user_id is None:
            return self.cb.recommend_tfidf(movie_title, top_n=top_n)

        # both empty
        if user_id is None and movie_title is None:
            return pd.DataFrame(columns=["title", "hybrid_score"])

        movie_title = str(movie_title).lower()

        # svd candidates movies
        svd_df = self.cf.recommend_svd(
            self.df,
            self.ratings,
            user_id=user_id,
            top_n=1000
        )

        svd_df = svd_df[svd_df["movie"].isin(self.df["title"])]

        candidate_movies = svd_df["movie"].tolist()

        # content score on candidate movies
        content_df = self.cb.tfidf_scores_for_candidates(
            movie_title,
            candidate_movies=candidate_movies,
            top_n=1000
        )

        # rename columns
        content_df = content_df.rename(columns={
            "movie": "title",
            "content_score": "content_score"
        })

        svd_df = svd_df.rename(columns={
            "movie": "title",
            "score": "svd_score"
        })

        # merge on title (inner join to keep only candidates with both scores)
        df = pd.merge(svd_df, content_df, on="title", how="inner")

        if df.empty:
            return pd.DataFrame(columns=["title", "hybrid_score"])

        # fill missing (safety)
        df["svd_score"] = df["svd_score"].fillna(0)
        df["content_score"] = df["content_score"].fillna(0)

        # normalize scores to 0-1 range
        scaler = MinMaxScaler()

        df[["svd_score", "content_score"]] = scaler.fit_transform(
            df[["svd_score", "content_score"]]
        )

        # hybrid score
        df["hybrid_score"] = (
            0.6 * df["svd_score"] +
            0.4 * df["content_score"]
        )

        return df.sort_values(
            "hybrid_score", ascending=False
        )[["title", "hybrid_score", "svd_score", "content_score"]].head(top_n).reset_index(drop=True)