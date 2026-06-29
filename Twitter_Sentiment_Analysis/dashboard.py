import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st


# -----------------------------
# Dataset Statistics
# -----------------------------
def show_statistics(df):

    st.subheader("📊 Dataset Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Tweets", len(df))
    col2.metric("Positive", (df["Sentiment"] == "Positive").sum())
    col3.metric("Negative", (df["Sentiment"] == "Negative").sum())
    col4.metric("Neutral", (df["Sentiment"] == "Neutral").sum())


# -----------------------------
# Pie Chart
# -----------------------------
def sentiment_pie_chart(df):

    st.subheader("🥧 Sentiment Distribution")

    sentiment_counts = df["Sentiment"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax.axis("equal")

    st.pyplot(fig)


# -----------------------------
# Bar Chart
# -----------------------------
def sentiment_bar_chart(df):

    st.subheader("📊 Sentiment Count")

    sentiment_counts = df["Sentiment"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))

    sentiment_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Tweets")
    ax.set_title("Sentiment Distribution")

    st.pyplot(fig)


