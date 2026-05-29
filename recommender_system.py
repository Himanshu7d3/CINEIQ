
from content_based_filtering import content_based_filtering
from collaborative_filtering import collaborative_filtering
class recommender_system:
    def __init__(self,df,ratings):
        self.df=df
        self.ratings=ratings
    def recommend_tfidf(self,movie_title):
        df=self.df
        cb=content_based_filtering()
        return cb.recommend_tfidf(df,movie_title)
    def recommend_svd(self,user_id):
        df=self.df
        ratings=self.ratings
        co=collaborative_filtering()
        return co.recommend_svd(df,ratings,user_id)