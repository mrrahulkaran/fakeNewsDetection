import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Define file paths
TRUE_PATH = "data/True.csv"
FAKE_PATH = "data/Fake.csv"
FEEDBACK_PATH = "data/feedback.csv"
MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Load and combine original data
def load_data():
    true_df = pd.read_csv(TRUE_PATH)
    fake_df = pd.read_csv(FAKE_PATH)

    true_df["label"] = "REAL"
    fake_df["label"] = "FAKE"

    df = pd.concat([true_df, fake_df], ignore_index=True)
    return df[["title", "text", "label"]]

# Load feedback data if available
def load_feedback_data():
    if os.path.exists(FEEDBACK_PATH) and os.path.getsize(FEEDBACK_PATH) > 0:
        return pd.read_csv(FEEDBACK_PATH)
    return pd.DataFrame(columns=["title", "text", "label"])

# Train the model and save it
def train_and_save():
    base_data = load_data()
    feedback_data = load_feedback_data()
    combined_data = pd.concat([base_data, feedback_data], ignore_index=True)

    combined_data.dropna(subset=["title", "text", "label"], inplace=True)
    combined_data["content"] = combined_data["title"] + " " + combined_data["text"]

    X = combined_data["content"]
    y = combined_data["label"]

    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    model = PassiveAggressiveClassifier(max_iter=1000)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(VECTORIZER_PATH, "wb") as vec_file:
        pickle.dump(tfidf_vectorizer, vec_file)

    accuracy = model.score(X_test, y_test)
    print(f"✅ Model trained and saved successfully. Accuracy: {accuracy:.2f}")

# Execute
if __name__ == "__main__":
    train_and_save()
