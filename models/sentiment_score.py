import numpy as np
import pandas as pd 
import json
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import os

class SentimentThresholdOptimizer:

    def __init__(self, sia, save_path=os.path.join("models","best_threshold.json")):
        self.save_path = save_path
        self.threshold = None
        self.sia = sia
        self.load()

    def clean_text(self, text):
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove stopwords
        stop_words = set(stopwords.words('english'))  # faster lookup
        words = [word for word in text.split() if word not in stop_words]

        return " ".join(words)
    
    # Train and update threshold (manual)
    def fit(self, df):
        cutpoint = []

        df1 = df.dropna().drop_duplicates().copy()
        df1['review'] = df1['review'].apply(self.clean_text)
        df1['compound'] = df1['review'].apply(lambda x: self.sia.polarity_scores(x)['compound'])

        for i in np.linspace(-1, 1, 40):
            df1['vader_sentiment'] = df1['compound'].apply(
                lambda x: "positive" if x > i else "negative"
            )
            accuracy = (df1['sentiment'] == df1['vader_sentiment']).mean()
            cutpoint.append((accuracy, i))

        best = max(cutpoint)
        self.threshold = best[1]

        self._save()
        print(f"Updated threshold: {self.threshold}")

        return self.threshold

    # Predict from compound score
    def predict_score(self, score):
        if self.threshold is None:
            raise ValueError("Threshold not available. Run fit() once.")
        return "positive" if score > self.threshold else "negative"
    
    def get_score(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return None

        return self.sia.polarity_scores(self.clean_text(text))['compound']
    
    # NEW: Predict directly from text
    def predict_sentiment(self, text):
        if self.threshold is None:
            raise ValueError("Threshold not available. Run fit() once.")

        if not isinstance(text, str) or text.strip() == "":
            return {"Sentiment": None, "Score": None, "Text": text}

        score = self.sia.polarity_scores(self.clean_text(text))['compound']
        sentiment = "positive" if score > self.threshold else "negative"

        return {
            "Sentiment": sentiment,
            "Score": score,
            "Text": text
        }

    # NEW: Batch prediction
    def predict_batch(self, texts):
        results = []
        for text in texts:
            results.append(self.predict_sentiment(text))
        
        df_results = pd.DataFrame(results)

        df_results['Score'] = (df_results['Score'].astype(float) + 1) / 2
        
        return df_results

    # Save threshold
    def _save(self):
        with open(self.save_path, "w") as f:
            json.dump({"threshold": self.threshold}, f)

    # Load threshold
    def load(self):
        try:
            with open(self.save_path, "r") as f:
                self.threshold = json.load(f)["threshold"]
        except:
            self.threshold = None      

def setup():
    df=pd.read_csv(os.path.join("Datasets","raw", "imdb-dataset-of-50k-movie-reviews", "IMDB Dataset.csv"))
    df['review'] = df['review'].str.replace(r'<br\s*/?>', '', regex=True)
    df['review'] = df['review'].str.lower()
    return df

class Sentiment:
    def __init__(self):
        self.df = setup()
        self.sia = SentimentIntensityAnalyzer()
        self.predictor = SentimentThresholdOptimizer(self.sia)

    def fit(self):
        print("Training model to find the best threshold. This may take a moment...")
        self.predictor.fit(self.df)

    def PredictScores(self, texts):
        # Check if the threshold was successfully loaded from the JSON file
        if self.predictor.threshold is None:
            print("No threshold file found! Triggering automatic fit()...")
            self.fit()
            
        df_results = self.predictor.predict_batch(texts)
        return np.round(df_results['Score'].astype(float), 3)