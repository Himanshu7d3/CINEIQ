import pandas as pd
import numpy as np
import string
# import nltk
# from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class content_based_filtering:
    

    def clean_text(self,text):
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)   # only remove extra spaces
        return text
        
    def make_new_df(self, df):

        df['tags'] = (

        (df['genres']) * 8 +

        (df['keywords']) * 5 +

        (df['Actors']) * 3 +

        (df['Director']) * 3 +
        df['tagline']+

        df['overview']

    )

        # Convert list to string
        # df['tags'] = df['tags'].apply(
        #     lambda x: " ".join(x)
        # )

        new_df = df[
            [
                'movieId',
                'imdbId',
                'tmdbId',
                'title',
                'genres',
                'keywords',
                'Actors',
                'overview',
                'Director',
                'tags'
            ]
        ]
        cols=['tags','genres','keywords','Actors','overview','Director']
        for item in cols:
            new_df[item]=new_df[item].apply(lambda x:' '.join(x))
            new_df[item]=new_df[item].apply(self.clean_text)

        new_df = new_df.drop_duplicates()
        new_df = new_df.drop_duplicates(subset='tmdbId')
        new_df=new_df.drop_duplicates(subset='title')

        return new_df
    
    def recommend_tfidf(self,df, movie_title, top_n=10):
        new_df=self.make_new_df(df)
        tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1,2),
    stop_words='english',
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)
        tfidf_matrix = tfidf.fit_transform(new_df['tags'])

        movie_title = movie_title.lower()

        # find movie
        match = new_df[new_df['title'].str.lower() == movie_title]

        if match.empty:
            print("Movie not found in dataset.")
            return pd.DataFrame()
        
        idx = match.index[0]

        # target movie info
        target_genres = set(
            str(new_df.loc[idx, 'genres']).lower().split()
        )

        target_actors = set(
            str(new_df.loc[idx, 'Actors']).lower().split()
        )

        target_director = str(new_df.loc[idx, 'Director']).lower()

        # TF-IDF vector for target movie
        target_vector = tfidf_matrix[idx]

        # similarity against ALL movies
        sim = cosine_similarity(tfidf_matrix, target_vector).flatten()

        scores = []

        for i in range(len(new_df)):

            if i == idx:
                continue

            score = sim[i]

            # GENRE BOOST 
            movie_genres = set(
                str(new_df.iloc[i]['genres']).lower().split()
            )

            genre_overlap = len(target_genres & movie_genres)

            score += genre_overlap * 0.08

            # ACTORS BOOST 
            movie_actors = set(
                str(new_df.iloc[i]['Actors']).lower().split()
            )

            cast_overlap = len(target_actors & movie_actors)

            score += cast_overlap * 0.03

            # DIRECTOR BOOST
            movie_director = str(new_df.iloc[i]['Director']).lower()

            if movie_director == target_director:
                score += 0.1

            scores.append((i, score))

        # sort
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        # top results
        movie_indices = [i[0] for i in scores[:top_n]]

        result = new_df.iloc[movie_indices][['title']].copy()

        result['score'] = [i[1] for i in scores[:top_n]]

        return result.reset_index(drop=True)

        