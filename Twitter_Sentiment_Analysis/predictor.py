import joblib
import os

from preprocessing import clean_text

# -----------------------------
# Load ML Models
# -----------------------------

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))


# -----------------------------
# Prediction Function
# -----------------------------

def predict_sentiment(text):

    # Clean the input
    cleaned = clean_text(text)

    # Convert into TF-IDF vector
    vector = tfidf.transform([cleaned])

    # Predict sentiment
    prediction = model.predict(vector)[0]

    # Prediction probability
    probability = model.predict_proba(vector)[0]

    confidence = max(probability) * 100

    sentiment = encoder.inverse_transform([prediction])[0]

    return sentiment, round(confidence, 2)



if __name__ == "__main__":
    sentiment, confidence = predict_sentiment("I absolutely love OpenAI!")
    print(sentiment, confidence)