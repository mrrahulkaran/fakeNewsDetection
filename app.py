# app.py

import streamlit as st
import joblib
import string
import numpy as np

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    return text

# Streamlit UI setup
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Paste a news article or headline to detect if it's **Fake** or **Real**.")

# Text input
user_input = st.text_area("Enter News Text (Max 500 characters)", max_chars=500, height=200)

# Simple keyword sanity lists
real_keywords = ['ministry', 'government', 'official', 'announced', 'award', 'released', 'filmfare', 'prime minister']
fake_keywords = ['shocking', 'click here', 'miracle', 'you won’t believe', 'hoax', 'conspiracy']

# On prediction button click
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some news content.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        proba = model.predict_proba(vectorized)[0]
        prediction = np.argmax(proba)
        confidence = round(np.max(proba) * 100, 2)

        fake_confidence = round(proba[1] * 100, 2)
        real_confidence = round(proba[0] * 100, 2)

        # Contextual words
        is_real_context = any(word in cleaned for word in real_keywords)
        is_fake_context = any(word in cleaned for word in fake_keywords)

        # Final decision with more trust threshold
        if fake_confidence >= 90:
            st.error(f"🔴 This news is **FAKE** (Confidence: {fake_confidence}%)")
        elif real_confidence >= 60:
            st.success(f"🟢 This news is **REAL** (Confidence: {real_confidence}%)")
        elif is_real_context and fake_confidence < 90:
            st.warning(f"⚠️ Possibly **REAL**, but model predicted **FAKE** (Confidence: {fake_confidence}%)")
        else:
            st.warning(f"⚠️ Uncertain. Model predicts FAKE with {fake_confidence}% confidence.")

