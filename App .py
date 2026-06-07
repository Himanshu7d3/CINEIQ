import streamlit as st
import pandas as pd
from recommender_system import recommender_system

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="CINEIQ AI",
    page_icon="🎬",
    layout="wide"
)

# ---------------- HEADER ----------------
st.title("🎬 CINEIQ AI Movie Recommendation System")
st.caption("Hybrid AI Engine: Collaborative + Content + Sentiment Based Recommendation")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    movies = pd.read_csv(r"Datasets\Cleaned\cleaned_data.csv")
    ratings = pd.read_csv(r"Datasets\Cleaned\final_ratings.csv")
    return movies, ratings

movies, ratings = load_data()

@st.cache_resource
def load_model():
    return recommender_system(movies, ratings)

recommend = load_model()

# ---------------- NAVIGATION ----------------
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "🎯 Recommender", "👥 Team"]
)

# =====================================================
# 🏠 HOME
# =====================================================
if page == "🏠 Home":

    st.subheader("🚀 Welcome")

    st.write(
        "CINEIQ is a hybrid recommendation system that suggests movies using AI techniques "
        "like SVD, TF-IDF and sentiment analysis."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Models Used", "3")
        st.write("Collaborative Filtering")

    with col2:
        st.metric("Tech Stack", "ML + Streamlit")
        st.write("Content-Based Filtering")

    with col3:
        st.metric("Engine Type", "Hybrid AI")
        st.write("Sentiment Boosting")

# =====================================================
# 🎯 RECOMMENDER
# =====================================================
elif page == "🎯 Recommender":

    st.subheader("🎯 Recommendation Engine")

    option = st.selectbox(
        "Choose Mode",
        ["User Recommendation", "Movie Recommendation", "Hybrid Recommendation"]
    )

    top_n = st.slider("Top N Results", 1, 20, 8)

    # ---------------- USER ----------------
    if option == "User Recommendation":

        user_id = st.number_input("Enter User ID", min_value=1, step=1)

        if st.button("Generate Recommendations"):

            result = recommend.hybrid_recommendation(
                user_id=user_id,
                top_n=top_n
            )

            st.success("Recommendations Ready")

            for _, row in result.iterrows():

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.subheader(row["title"])
                    st.write(f"SVD Score: {round(row['svd_score'], 4)}")
                    st.write(f"Sentiment Score: {round(row['sentiment_score'], 4)}")

                with col2:
                    st.metric("Hybrid Score", round(row["hybrid_score"], 4))
                    st.progress(min(float(row["hybrid_score"]), 1.0))

                st.divider()

    # ---------------- MOVIE ----------------
    elif option == "Movie Recommendation":

        movie_title = st.text_input("Enter Movie Title")

        if st.button("Find Similar Movies"):

            result = recommend.hybrid_recommendation(
                movie_title=movie_title,
                top_n=top_n
            )

            st.success("Similar Movies Found")

            for _, row in result.iterrows():

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.subheader(row["title"])
                    st.write(f"Content Score: {round(row['content_score'], 4)}")
                    st.write(f"Sentiment Score: {round(row['sentiment_score'], 4)}")

                with col2:
                    st.metric("Score", round(row["hybrid_score"], 4))
                    st.progress(min(float(row["hybrid_score"]), 1.0))

                st.divider()

    # ---------------- HYBRID ----------------
    else:

        user_id = st.number_input("Enter User ID", min_value=1, step=1)
        movie_title = st.text_input("Enter Movie Title")

        if st.button("Get Best Recommendations"):

            result = recommend.hybrid_recommendation(
                user_id=user_id,
                movie_title=movie_title,
                top_n=top_n
            )

            st.success("Hybrid Recommendations Generated")

            for _, row in result.iterrows():

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.subheader(row["title"])
                    st.write(f"SVD Score: {round(row['svd_score'], 4)}")
                    st.write(f"Content Score: {round(row['content_score'], 4)}")
                    st.write(f"Sentiment Score: {round(row['sentiment_score'], 4)}")

                with col2:
                    st.metric("Final Score", round(row["hybrid_score"], 4))
                    st.progress(min(float(row["hybrid_score"]), 1.0))

                st.divider()

# =====================================================
# 👥 TEAM
# =====================================================
elif page == "👥 Team":

    st.subheader("👥 Project Team")

    st.write("""
    - Khusi Patra — ML Model + Backend
    - Member 2 — Data Engineering
    - Member 3 — UI Development
    - Member 4 — Testing & Documentation
    """)

    st.info("CINEIQ Hybrid Recommendation System 🚀")