import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from models.explainability import Explainer
from models.recommender_system import recommender_system

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CINEIQ – Movie Recommendations",
    page_icon="🎬",
    layout="wide"
)

# =====================================================
# DATA LOADING  (cached – runs only once per session)
# =====================================================
@st.cache_data
def load_data():
    movies  = pd.read_csv("Datasets/Cleaned/cleaned_data.csv")
    ratings = pd.read_csv("Datasets/Cleaned/final_ratings.csv")
    return movies, ratings


movies, ratings = load_data()


# =====================================================
# MODEL LOADING  (cached resource – pkl files reused,
#                 no retraining at runtime)
# =====================================================
@st.cache_resource
def load_model(_movies, _ratings):
    return recommender_system(_movies, _ratings)


recommend = load_model(movies, ratings)
explainer = Explainer(movies)


# =====================================================
# HEADER
# =====================================================
st.title("🎬 CINEIQ – Movie Recommendation System")
st.caption(
    "Hybrid AI engine · Collaborative Filtering · Content-Based · Sentiment-Aware"
)

# =====================================================
# SIDEBAR – navigation
# =====================================================
st.sidebar.title("CINEIQ")
page = st.sidebar.radio("Navigate", [" Home", " Recommender", " My Taste"])


# =====================================================
# HOME PAGE
# =====================================================
if page == " Home":

    st.subheader("Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Movies",  "42,278")
    c2.metric("Total Users",   "270,896")
    c3.metric("Engine",        "Hybrid AI")

    st.divider()
    st.write(
        """
        CINEIQ is a hybrid movie recommendation system combining:

        - **Collaborative Filtering** (SVD) – personalised picks based on user history  
        - **Content-Based Filtering** (TF-IDF) – similar movies by cast, genre, keywords  
        - **Sentiment Scoring** (VADER) – boosts highly-reviewed titles  

        """
    )


# =====================================================
# RECOMMENDER PAGE
# =====================================================
elif page == " Recommender":

    st.subheader("Recommendation Engine")

    # ── Sidebar controls ──────────────────────────────
    mode = st.sidebar.selectbox(
        "Mode",
        ["User Recommendation", "Movie Recommendation", "Hybrid Recommendation"],
    )

    top_n = st.sidebar.slider("Results", min_value=5, max_value=20, value=10)

    user_id     = None
    movie_title = None

    if mode == "User Recommendation":
        user_id = st.sidebar.number_input("User ID", min_value=1, step=1)
        run = st.sidebar.button("Generate Recommendations")

    elif mode == "Movie Recommendation":
        movie_title = st.sidebar.text_input("Movie Title")
        run = st.sidebar.button("Find Similar Movies")

    else:  # Hybrid
        user_id     = st.sidebar.number_input("User ID", min_value=1, step=1)
        movie_title = st.sidebar.text_input("Movie Title")
        run = st.sidebar.button("Get Recommendations")

    # ── Movie card ────────────────────────────────────
    # ave_rating and genres come directly from the recommender
    # result (already stored in cleaned_data.csv) – no extra
    # lookup or merge needed at display time.
    def show_movie(row, rank: int, explanation: str = ""):
        genres     = row.get("genres",     "Unknown") or "Unknown"
        ave_rating = row.get("ave_rating", 0.0)

        if pd.isna(genres):     genres = "Unknown"
        if pd.isna(ave_rating): ave_rating = 0.0

        st.caption(f"#{rank}")
        info_col, score_col = st.columns([6, 1])

        with info_col:
            st.subheader(row["title"])
            st.write(f"⭐ Avg Rating : {round(float(ave_rating), 2)}")
            st.write(f"🎭 Genres     : {genres}")
            if explanation:
                st.caption(f"💡 Why: {explanation}")

        with score_col:
            st.metric("Score", round(float(row["hybrid_score"]), 4))

        st.divider()

    # ── Run recommendation ────────────────────────────
    if run:
        try:
            if mode == "User Recommendation":
                result = recommend.hybrid_recommendation(
                    user_id=user_id, top_n=top_n
                )

            elif mode == "Movie Recommendation":
                result = recommend.hybrid_recommendation(
                    movie_title=movie_title.strip(), top_n=top_n
                )

            else:
                result = recommend.hybrid_recommendation(
                    user_id=user_id,
                    movie_title=movie_title.strip(),
                    top_n=top_n,
                )

            if result.empty:
                st.warning("No recommendations found. Try different inputs.")
            else:
                st.success(f"{len(result)} recommendations ready.")
                for rank, (_, row) in enumerate(result.iterrows(), start=1):
                    explanation = explainer.explain(
                        rec_title     = row["title"],
                        source_title  = movie_title if movie_title else None,
                        svd_score     = row.get("svd_score"),
                        content_score = row.get("content_score")
                    )
                    show_movie(row, rank, explanation)

        except Exception as e:
            st.error(f"Error: {e}")
    
elif page == " My Taste":
    st.subheader("My Taste Dashboard")
    uid = st.sidebar.number_input("User ID", min_value=1, step=1, key="dash_uid")
    run_dash = st.sidebar.button("Load Dashboard")
    if run_dash:
        user_ratings = ratings[ratings['userId'] == uid]
        if user_ratings.empty:
            st.warning("No ratings found for this user.")
        else:
            user_movies = user_ratings.merge(movies, on='movieId', how='inner')
            sns.set_theme(style="darkgrid")
            # ── 1. Genre Radar Chart ──────────────────
            st.markdown("### 🎭 Genre Preferences")
            genre_rows = []
            for _, row in user_movies.iterrows():
                for g in str(row['genres']).split():
                    if g and g != 'nan':
                        genre_rows.append({'genre': g.title(), 'rating': row['rating']})

            if genre_rows:
                genre_df = (pd.DataFrame(genre_rows)
                            .groupby('genre')['rating']
                            .mean()
                            .sort_values(ascending=False)
                            .head(8)     # keep 8 for clean radar
                            .reset_index())
                labels  = genre_df['genre'].tolist()
                values  = genre_df['rating'].tolist()
                N       = len(labels)
                # angles for each spoke — close the circle by repeating first angle
                angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
                values  += [values[0]]
                angles  += [angles[0]]
                labels  += [labels[0]]

                fig, ax = plt.subplots(figsize=(5, 5),
                                    subplot_kw=dict(polar=True),
                                    facecolor='#0e1117')
                ax.set_facecolor('#0e1117')
                ax.plot(angles, values, color='royalblue', linewidth=2)
                ax.fill(angles, values, color='royalblue', alpha=0.25)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(labels[:-1], color='white', fontsize=9)
                ax.set_ylim(0, 5)
                ax.yaxis.set_tick_params(labelcolor='grey')
                ax.spines['polar'].set_color('grey')
                ax.grid(color='grey', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # ── 2. Decade Preferences ─────────────────
            st.markdown("### 📅 Decade Preferences")
            if 'decade' in user_movies.columns:
                decade_df = (user_movies.dropna(subset=['decade'])
                                        .groupby('decade')['rating']
                                        .mean()
                                        .reset_index()
                                        .sort_values('decade'))
                decade_df['decade'] = decade_df['decade'].astype(int).astype(str) + 's'

                fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0e1117')
                ax.set_facecolor('#0e1117')
                sns.barplot(data=decade_df, x='decade', y='rating',
                            palette='Blues_d', ax=ax)
                ax.set_xlabel("Decade", color='white')
                ax.set_ylabel("Avg Rating", color='white')
                ax.tick_params(colors='white')
                ax.set_ylim(0, 5)

                for spine in ax.spines.values():
                    spine.set_edgecolor('grey')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Decade data not available — delete cleaned_data.csv and re-run cleaning.")
            # ── 3. Director Affinities ────────────────
            st.markdown("### 🎥 Favourite Directors")

            dir_rows = []
            for _, row in user_movies.iterrows():
                for d in str(row['Director']).split():
                    if d and d != 'nan':
                        dir_rows.append({'director': d.title(), 'rating': row['rating']})
            if dir_rows:
                dir_df = (pd.DataFrame(dir_rows)
                            .groupby('director')['rating']
                            .agg(['mean', 'count'])
                            .reset_index()
                            .rename(columns={'mean': 'avg_rating', 'count': 'movies_rated'})
                            .query('movies_rated >= 2')
                            .sort_values('avg_rating', ascending=False)
                            .head(10))
                if not dir_df.empty:
                    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0e1117')
                    ax.set_facecolor('#0e1117')
                    sns.barplot(data=dir_df, x='avg_rating', y='director',
                                palette='Greens_d', ax=ax)
                    ax.set_xlabel("Avg Rating", color='white')
                    ax.set_ylabel("", color='white')
                    ax.tick_params(colors='white')
                    ax.set_xlim(0, 5)
                    
                    for spine in ax.spines.values():
                        spine.set_edgecolor('grey')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Not enough data — need at least 2 rated movies per director.")
            # ── 4. Actor Affinities ───────────────────
            st.markdown("### 🎬 Favourite Actors")
            actor_rows = []
            for _, row in user_movies.iterrows():
                for a in str(row['Actors']).split():
                    if a and a != 'nan':
                        actor_rows.append({'actor': a.title(), 'rating': row['rating']})
            if actor_rows:
                actor_df = (pd.DataFrame(actor_rows)
                            .groupby('actor')['rating']
                            .agg(['mean', 'count'])
                            .reset_index()
                            .rename(columns={'mean': 'avg_rating', 'count': 'movies_rated'})
                            .query('movies_rated >= 2')
                            .sort_values('avg_rating', ascending=False)
                            .head(10))
                if not actor_df.empty:
                    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#0e1117')
                    ax.set_facecolor('#0e1117')
                    sns.barplot(data=actor_df, x='avg_rating', y='actor',
                                palette='Oranges_d', ax=ax)
                    ax.set_xlabel("Avg Rating", color='white')
                    ax.set_ylabel("", color='white')
                    ax.tick_params(colors='white')
                    ax.set_xlim(0, 5)
                    for spine in ax.spines.values():
                        spine.set_edgecolor('grey')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Not enough data — need at least 2 rated movies per actor.")