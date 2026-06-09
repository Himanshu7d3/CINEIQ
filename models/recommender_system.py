import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from models.content_based_filtering import ContentBasedFiltering
from models.collaborative_filtering import collaborative_filtering


class recommender_system:

    def __init__(self, df, ratings):
        self.df = df.copy()
        self.ratings = ratings

        self.cb = ContentBasedFiltering(self.df)
        self.cf = collaborative_filtering()

    def hybrid_recommendation(self, user_id=None, movie_title=None, top_n=10):

        sentiment_df = self.df[['title', 'sentiment_score','ave_rating','genres']].drop_duplicates()
        scaler = MinMaxScaler()

        # only user_id provided, no movie_title
        if user_id is not None and movie_title is None:
            svd_df = self.cf.recommend_svd(
                self.df, self.ratings, user_id=user_id, top_n=1000
            )
            svd_df = svd_df[svd_df["movie"].isin(self.df["title"])]

            svd_df = svd_df.rename(columns={"movie": "title", "score": "svd_score"})
            svd_df = pd.merge(svd_df, sentiment_df, on="title", how="left")        

            # normalize scores to 0-1 range
            svd_df[["svd_score", "sentiment_score"]] = scaler.fit_transform(
                svd_df[["svd_score", "sentiment_score"]]
            )

            # hybrid score
            svd_df["hybrid_score"] = (
                0.75 * svd_df["svd_score"] +
                0.25 * svd_df["sentiment_score"]
            )

            svd_df["content_score"] = 0.0
            return svd_df.sort_values("hybrid_score", ascending=False)[
                ["title", "hybrid_score", "svd_score", "content_score", "ave_rating", "genres"]
            ].head(top_n).reset_index(drop=True)

        # only movie_title provided, no user_id
        if movie_title is not None and user_id is None:
            df = self.cb.recommend_tfidf(movie_title, top_n=1000)
            if df.empty:
                    return pd.DataFrame()

            df = df.rename(columns={"movie": "title", "score": "content_score"})
            df = pd.merge(df, sentiment_df, on="title", how="left")
            df["sentiment_score"] = df["sentiment_score"].fillna(0)

            # normalize scores to 0-1 range
            df[["content_score", "sentiment_score"]] = scaler.fit_transform(
                df[["content_score", "sentiment_score"]]
            )

            # hybrid score
            df["hybrid_score"] = (
                0.75 * df["content_score"] +
                0.25 * df["sentiment_score"]
            )

            df["svd_score"] = 0.0
            return df.sort_values("hybrid_score", ascending=False)[
                ["title", "hybrid_score", "svd_score", "content_score", "ave_rating", "genres"]
            ].head(top_n).reset_index(drop=True)

        # both empty
        if user_id is None and movie_title is None:
            return pd.DataFrame(columns=["title", "hybrid_score"])

        movie_title = str(movie_title).lower()

        # svd candidates movies
        svd_df = self.cf.recommend_svd(
            self.df, self.ratings, user_id=user_id, top_n=1000
        )
        svd_df = svd_df[svd_df["movie"].isin(self.df["title"])]
        candidate_movies = svd_df["movie"].tolist()

        # content score on candidate movies
        content_df = self.cb.tfidf_scores_for_candidates(
            movie_title, candidate_movies=candidate_movies, top_n=1000
        )

        content_df = content_df.rename(columns={"movie": "title", "score": "content_score"})
        svd_df = svd_df.rename(columns={"movie": "title", "score": "svd_score"})

        # merge on title
        df = pd.merge(svd_df, content_df, on="title", how="inner")

        if df.empty:
            return pd.DataFrame(columns=["title", "hybrid_score"])

        df = pd.merge(df, sentiment_df, on="title", how="left")

        # fill missing
        df["svd_score"] = df["svd_score"].fillna(0)
        df["content_score"] = df["content_score"].fillna(0)

        # normalize all scores to 0-1 range
        df[["svd_score", "content_score", "sentiment_score"]] = scaler.fit_transform(
            df[["svd_score", "content_score", "sentiment_score"]]
        )

        # hybrid score
        df["hybrid_score"] = (
            0.50 * df["svd_score"] +
            0.35 * df["content_score"] +
            0.15 * df["sentiment_score"]
        )

        return df.sort_values("hybrid_score", ascending=False)[
        ['title', 'hybrid_score', 'svd_score', 'content_score', 'ave_rating', 'genres']].head(top_n).reset_index(drop=True)