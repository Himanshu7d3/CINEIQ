import pandas as pd
import numpy as np
import ast

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
        cleaned_pro=[]

        for row in df['crew']:
            directors = []
            producer=[]

            if isinstance(row, list):
                for item in row:
                    if item.get('department') == 'directing':
                        name = item.get('name', '')
                        directors.append(name.replace(" ", "").lower())
                    if item.get('department')=='production':
                        name=item.get('name','')
                        producer.append(name.replace(" ","").lower())

            cleaned_dir.append(directors)
            cleaned_pro.append(producer)
        df['Director'] = cleaned_dir
        df['Producer'] = cleaned_pro
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
    
    def clean_belongs_to_collection(self,df):

        cleaned = []

        for item in df['belongs_to_collection']:

            if isinstance(item, dict):

                name = item.get('name', '')
                name = name.replace(" ", "").lower()

                cleaned.append([name])   # LIST

            else:
                cleaned.append([])

        df['belongs_to_collection'] = cleaned

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
        df['tagline']=df['tagline'].apply(lambda x:x.split())
        return df
    def rating_transform(self,df):
        new_rating=df[['rating','movieId']]
        df2=new_rating.groupby('movieId').sum().reset_index()
        df3=new_rating['movieId'].value_counts().reset_index(name='counts')
        df2=df2.merge(df3,on='movieId')
        df2['ave_rating']=df2['rating']/df2['counts']
        return df2
    
    def clean(self, df):

        # Remove duplicates
        df = df.drop_duplicates()

        # Lowercase titles
        # df['title'] = df['title'].str.lower()

        df=self.clean_imdb(df)
        df.drop(columns=['imdb_id'], inplace=True)

        df=self.clean_tmdbId(df)
        df.drop(columns=['id'], inplace=True)

        col_name=['cast','crew','keywords','belongs_to_collection','spoken_languages','production_companies','genres','production_countries']
        for item in col_name:
            df[item] = df[item].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else []) 

        df = self.clean_cast(df) 
        df=self.clean_crew(df)
        df= self.clean_keywords(df)
        df=self.clean_belongs_to_collection(df)
        df=self.clean_overview(df)
        df=self.clean_genres(df)
        df=self.clean_tagline(df)

        return df
