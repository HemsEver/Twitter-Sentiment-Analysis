import streamlit as st
import pandas as pd
from predictor import predict_sentiment
from dashboard import (
    show_statistics,
    sentiment_pie_chart,
    sentiment_bar_chart,
    
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Powered Twitter Sentiment Analysis",
    page_icon="😊",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(
    "data/twitter_training.csv",
    header=None,
    names=["ID", "Topic", "Sentiment", "Tweet"]
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🤖 Twitter Sentiment Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🤖 Predict",
        "☁️ Word Cloud",
        "ℹ️ About"
    ]
)

# -----------------------------
# HOME
# -----------------------------
if page == "🏠 Home":

    st.title("😊 AI Powered Twitter Sentiment Analysis")

    st.markdown("---")

    st.write("""
Welcome!

This application predicts whether a tweet is

- 😊 Positive
- 😐 Neutral
- 😡 Negative

using **Natural Language Processing (NLP)** and **Machine Learning**.
""")

    st.subheader("📌 Technologies Used")

    st.write("""
- Python
- Streamlit
- Pandas
- Scikit-Learn
- NLTK
- TF-IDF Vectorizer
- Logistic Regression
- Naive Bayes
""")

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

# -----------------------------
# DASHBOARD
# -----------------------------
elif page == "📊 Dashboard":

    st.title("📊 Dashboard")

    show_statistics(df)

    st.markdown("---")

    sentiment_pie_chart(df)

    st.markdown("---")

    sentiment_bar_chart(df)

# -----------------------------
# PREDICTION
# -----------------------------
elif page == "🤖 Predict":

    st.title("🤖 Predict Tweet Sentiment")

    tweet = st.text_area(
        "Enter a Tweet",
        height=150,
        placeholder="Type your tweet here..."
    )

    if st.button("Predict Sentiment"):

        if tweet.strip() == "":
            st.warning("Please enter a tweet.")

        else:

            sentiment, confidence = predict_sentiment(tweet)

            st.success(f"Prediction : {sentiment}")

            st.info(f"Confidence : {confidence}%")



# -----------------------------
# ABOUT
# -----------------------------
elif page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.write("""
### Project Title

AI Powered Twitter Sentiment Analysis using NLP & Machine Learning

---

### Objective

To classify tweets into

- Positive
- Neutral
- Negative

using Machine Learning algorithms.

---

### Algorithms Used

- Logistic Regression
- Naive Bayes

---

### NLP Techniques

- Lowercasing
- Stopword Removal
- Lemmatization
- TF-IDF Vectorization

---

### Developed By

Hemavathi B

### Future Scope

- Live Twitter API Integration
- Deep Learning Models (LSTM/BERT)
- Real-Time Sentiment Dashboard
- Multi-language Sentiment Analysis
""")