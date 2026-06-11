# 🎬 CINEIQ — Explainable Movie Recommendation Engine

> A hybrid movie recommendation system combining Collaborative Filtering, Content-Based Filtering, and Sentiment Analysis — with human-readable explanations for every recommendation.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Author](#author)

---

## Overview

Content discovery on modern streaming platforms is opaque, biased toward promoted titles, and traps users in recommendation loops. **CINEIQ** is an open, explainable movie recommendation engine that combines multiple ML strategies to deliver personalised, interpretable suggestions.

---

## Features

- 🤝 **Collaborative Filtering** — SVD-based matrix factorisation on user rating history
- 📄 **Content-Based Filtering** — TF-IDF + cosine similarity on cast, genres, keywords, overview, tagline
- 💬 **Sentiment-Aware Re-Ranking** — VADER sentiment scores from movie overviews and taglines boost well-received titles
- 🔀 **Hybrid Engine** — weighted ensemble of all three signals
- 💡 **Explainability Layer** — every recommendation shows a human-readable reason (shared genre, same director, top-rated, etc.)
- 📊 **User Taste Dashboard** — genre radar chart, decade preferences, director and actor affinities from a user's rating history
- ⚡ **Fast Inference** — pickle-cached TF-IDF and SVD models (~0.7s per recommendation)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.9+ |
| ML / Data | scikit-learn, NumPy, pandas, SciPy |
| NLP | VADER Sentiment (vaderSentiment), NLTK |
| Visualisation | Matplotlib, Seaborn |
| App | Streamlit |
| Data Source | KaggleHub |

---

## Project Structure

```
CINEIQ/
│
├── App.py                             # Streamlit web application
├── main.ipynb                         # Main notebook — full pipeline command centre
├── train_models.py                    # Builds and saves TF-IDF and SVD pickle files
├── requirements.txt                   # All dependencies
│
├── models/
│   ├── recommender_system.py          # Hybrid recommendation engine
│   ├── content_based_filtering.py     # TF-IDF + cosine similarity
│   ├── collaborative_filtering.py     # SVD matrix factorisation
│   ├── sentiment_score.py             # VADER sentiment scoring + threshold optimiser
│   └── explainability.py              # Rule-based explanation generator
│
├── Datasets/
│   ├── Raw/                           # Downloaded Kaggle datasets (git-ignored)
│   ├── Cleaned/                       # Processed CSVs (git-ignored)
│   │    ├── cleaned_data.csv
│   │    └── final_ratings.csv
│   │
│   ├── data_cleaning.py               # Full data cleaning pipeline
│   └── dataset_download.py            # Kaggle dataset downloader
│
├── tfidf_data.pkl                     # Cached TF-IDF model (auto-generated)
└── svd_data.pkl                       # Cached SVD model (auto-generated)
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone <repository_url>
cd CINEIQ
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Kaggle API credentials

CINEIQ downloads datasets from Kaggle automatically. You need a Kaggle account and API key.

1. Go to [kaggle.com](https://www.kaggle.com) → Account → API → **Create Legacy API Key**
2. This downloads a `kaggle.json` file
3. Place it at:
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
   - **Mac/Linux:** `~/.kaggle/kaggle.json`

### 5. Download NLTK data (one-time)

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 6. Run the full pipeline

Open `main.ipynb` in Jupyter and run all cells top to bottom. This will:

1. Download all datasets from Kaggle
2. Merge and clean the data
3. Train and cache TF-IDF and SVD models
4. Save cleaned CSVs to `Datasets/Cleaned/`

### 7. Launch the app

```bash
streamlit run App.py
```

---

## How It Works

### Recommendation Modes

| Mode | Input | Signal Used |
|---|---|---|
| **Movie Recommendation** | Movie title | TF-IDF content score + sentiment |
| **User Recommendation** | User ID | SVD collaborative score + sentiment |
| **Hybrid Recommendation** | User ID + Movie title | SVD + TF-IDF + sentiment (weighted ensemble) |

### Hybrid Score Formula

```
# Movie-only
hybrid_score = 0.75 × content_score + 0.25 × sentiment_score

# User-only
hybrid_score = 0.75 × svd_score + 0.25 × sentiment_score

# Both
hybrid_score = 0.50 × svd_score + 0.35 × content_score + 0.15 × sentiment_score
```

### Explainability

Each recommendation card shows a reason such as:

```
💡 Why: 🎭 Matches genre: Action, Thriller  ·  👥 Users like you loved this  ·  🌟 Highly praised by audiences
```

---

## Dataset

| Dataset | Source | Used For |
|---|---|---|
| The Movies Dataset | [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) | Movie metadata, ratings |
| IMDB 50K Reviews | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | Sentiment threshold training |

> **Note:** Datasets are downloaded automatically via KaggleHub. Raw and cleaned data folders are git-ignored.

---

## Author

**Himanshu || Khusi || Prithwish**

---

