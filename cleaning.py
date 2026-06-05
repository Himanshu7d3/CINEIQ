import re
import os
import pandas as pd
import numpy as np
import ast
import nltk
from SentimentScore import Sentiment 

class DataCleaner:
    def clean_tmdbId(self,df):
        # rows where tmdbId is missing
        # but id exists
        mask = (
            df['tmdbId'].isnull() &
            df['id'].notnull()
        )

        # fill tmdbId using id
        df.loc[mask, 'tmdbId'] = df.loc[mask, 'id']

        # convert tmdbId to nullable integer
        df['tmdbId'] = df['tmdbId'].astype('Int64')

        return df   

    def clean_imdb(self, df):


    # rows where imdbId is missing
    # but imdb_id exists
        mask = (
        df['imdbId'].isnull() &
        df['imdb_id'].notnull()
    )

    # fill imdbId using imdb_id
        df.loc[mask, 'imdbId'] = (
        df.loc[mask, 'imdb_id']
        .astype(str)
        .str.replace('tt', '', regex=False)
        .astype(float)
    )

    # convert imdbId to nullable integer
        df['imdbId'] = df['imdbId'].astype('Int64')

        return df

    def clean_cast(self, df):

        cleaned_cast = []

        for row in df['cast']:

            actors = []
            counter = 0

            if isinstance(row, list):

                for item in row:

                    if counter < 5:

                        actors.append(
                            item.get('name', '')
                            .replace(" ", "")
                            .lower()
                        )

                        counter += 1

            cleaned_cast.append(actors)

        df['Actors'] = cleaned_cast

        return df
    def clean_crew(self,df):
        new_col = []

        for lst in df['crew']:

            if isinstance(lst, list):

                temp = []

                for d in lst:

                    if 'department' in d:
                        d['department'] = d['department'].lower()

                    temp.append(d)

                new_col.append(temp)

            else:
                new_col.append(lst)

        df['crew'] = new_col
        cleaned_dir = []

        for row in df['crew']:
            directors = []

            if isinstance(row, list):
                for item in row:
                    if item.get('department') == 'directing':
                        name = item.get('name', '')
                        directors.append(name.replace(" ", "").lower())

            cleaned_dir.append(directors)
        df['Director'] = cleaned_dir
        return df

    def clean_keywords(self,df):
        cleaned = []
        
        for row in df['keywords']:
            words = []

            if isinstance(row, list):
                for item in row:
                    if isinstance(item, dict) and 'name' in item: 
                        words.append(item['name'].replace(" ", "").lower())
            
            cleaned.append(words)
        
        df['keywords'] = cleaned
        return df
    
    def clean_overview(self,df):

    # lowercase and strip spaces
        df['overview'] = (
            df['overview']
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # replace invalid text
        df['overview'] = df['overview'].replace(
            ['no overview found.', 'no overview'],
            np.nan
        )

        # replace released with NaN
        df['overview'] = df['overview'].replace(
            'released',
            np.nan
        )

        # fill missing overview with empty string
        df['overview'] = df['overview'].fillna('')

        # for sentiment score calulation
        df['sentiment']=df['overview']

        # split into words
        df['overview'] = df['overview'].apply(
            lambda x: x.split()
        )

        return df
    def clean_genres(self,df):
    
        cleaned_genres = []
        
        for row in df['genres']:
            
            genres = []
            
            if isinstance(row, list):
                
                for item in row:
                    
                    if isinstance(item, dict):
                        
                        genres.append(item.get('name', '').replace(" ", "").lower())
            
            cleaned_genres.append(genres)
        
        df['genres'] = cleaned_genres
        
        return df
    def clean_tagline(self,df):
        df['tagline']=df['tagline'].fillna('')
        df['tagline']=df['tagline'].str.lower()
        # concatenate in sentiment column for sentment score
        df['sentiment']=df['sentiment']+df['tagline']
        df['tagline']=df['tagline'].apply(lambda x:x.split())
        return df
    def clean_text(self,text):
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)   # only remove extra spaces
        return text
    def clean_columns(self,df):
        col=['cast','crew','adult','budget','homepage','original_language','original_title','popularity','poster_path','production_companies','production_countries',
              'release_date','revenue','runtime','spoken_languages','status','video','vote_average','vote_count','belongs_to_collection']
        df.drop(columns=col,inplace=True)
        df['tags'] = (

        (df['genres']) * 8 +

        (df['keywords']) * 5 +

        (df['Actors']) * 3 +

        (df['Director']) * 3 +
        df['tagline']+
        df['overview']
        )
        cols=['tags','genres','keywords','Actors','overview','Director','tagline']
        for item in cols:
            df[item]=df[item].apply(lambda x:' '.join(x))
            df[item]=df[item].apply(self.clean_text)

        df = df.drop_duplicates()
        df = df.drop_duplicates(subset='tmdbId')
        df=df.drop_duplicates(subset='title')
        df=df.reset_index(drop=True)
        return df
    
    def setimentcsore(self,df):
        ss=Sentiment()
        df['sentiment_score']=ss.PredictScores(df['sentiment'])
        df.drop('sentiment',axis=1,inplace=True)
        return df

    def clean(self, df=None, combined_ratings=None):

        save_dir = os.path.join('Datasets', 'Cleaned')
        save_path = os.path.join(save_dir, 'cleaned_data.csv')
        save_ratings_path = os.path.join(save_dir, 'final_ratings.csv')

        # Load cached files if they already exist
        if os.path.exists(save_path) and os.path.exists(save_ratings_path):

            cleaned_df = pd.read_csv(save_path)
            ratings_df = pd.read_csv(save_ratings_path)

            return cleaned_df, ratings_df

        # Input validation
        if df is None or combined_ratings is None:
            raise ValueError(
                "df and combined_ratings must be provided when cache files do not exist."
            )

        os.makedirs(save_dir, exist_ok=True)

        print("Starting data cleaning process...")

        # Remove duplicates
        df = df.drop_duplicates()

        # Lowercase titles
        df['tf_idf_title'] = df['title'].str.lower()

        df = self.clean_imdb(df)
        df.drop(columns=['imdb_id'], inplace=True)

        df = self.clean_tmdbId(df)
        df.drop(columns=['id'], inplace=True)

        col_name = [
            'cast',
            'crew',
            'keywords',
            'belongs_to_collection',
            'genres'
        ]

        for item in col_name:
            df[item] = df[item].apply(
                lambda x: ast.literal_eval(x) if pd.notnull(x) else []
            )

        df = self.clean_cast(df)
        df = self.clean_crew(df)
        df = self.clean_keywords(df)
        df = self.clean_overview(df)
        df = self.clean_genres(df)
        df = self.clean_tagline(df)
        df = self.setimentcsore(df)
        df = self.clean_columns(df)

        # Save cleaned files
        df.to_csv(save_path, index=False)
        combined_ratings.to_csv(save_ratings_path, index=False)

        return df, combined_ratings