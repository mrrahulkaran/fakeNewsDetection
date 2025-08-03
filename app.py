import streamlit as st
import pandas as pd
import joblib
import os

# Set page configuration
st.set_page_config(page_title="🕵️ Fake News Detector", page_icon="📰", layout="centered")

# File paths
MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
FEEDBACK_PATH = "data/feedback.csv"

# Load model and vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# App Header
st.markdown("""
    <h1 style='text-align: center;'>🕵️‍♂️ Fake News Detector</h1>
    <p style='text-align: center;'>Predict whether a news article is <b>FAKE</b> or <b>REAL</b>.</p>
""", unsafe_allow_html=True)

# News Input Form
with st.form(key="prediction_form"):
    st.subheader("📰 Enter News Details")
    title = st.text_input("News Title", placeholder="Enter the news headline...")
    body = st.text_area("News Body", placeholder="Paste the full news article here...", height=200)
    predict_button = st.form_submit_button("🚀 Predict")

# Prediction Logic
if predict_button:
    if not title.strip() or not body.strip():
        st.warning("⚠️ Please fill in both the title and the body of the news.")
    else:
        combined_text = f"{title.strip()} {body.strip()}"

        # Check for feedback override
        label_from_feedback = None
        if os.path.exists(FEEDBACK_PATH):
            try:
                feedback_df = pd.read_csv(FEEDBACK_PATH)
                match = feedback_df[
                    (feedback_df["title"] == title.strip()) &
                    (feedback_df["text"] == body.strip())
                ]
                if not match.empty:
                    label_from_feedback = match.iloc[-1]["label"]
            except Exception as e:
                st.error(f"Error reading feedback: {e}")

        # Use feedback label if available
        if label_from_feedback:
            if label_from_feedback == "FAKE":
                st.error("🛑 This news is marked as **FAKE** based on previous feedback.")
            else:
                st.success("✅ This news is marked as **REAL** based on previous feedback.")
        else:
            # Model prediction
            transformed_text = vectorizer.transform([combined_text])
            prediction = model.predict(transformed_text)[0]

            if prediction == "FAKE":
                st.error("🛑 This news is predicted to be **FAKE**.")
            else:
                st.success("✅ This news is predicted to be **REAL**.")

# Feedback Submission
with st.expander("💬 Submit Feedback (optional)"):
    with st.form("feedback_form"):
        user_feedback = st.radio("How would you label this news?", ["REAL", "FAKE"])
        submit_feedback = st.form_submit_button("✅ Submit Feedback")

        if submit_feedback:
            feedback_entry = pd.DataFrame([[title.strip(), body.strip(), user_feedback]], columns=["title", "text", "label"])
            try:
                if os.path.exists(FEEDBACK_PATH):
                    feedback_entry.to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
                else:
                    feedback_entry.to_csv(FEEDBACK_PATH, index=False)
                st.success("🎉 Thank you! Your feedback has been recorded.")
            except Exception as e:
                st.error(f"Error saving feedback: {e}")

    # ✅ Safe: Add Home button *outside* the form
    # ✅ Home button (safe placement outside form)
if st.button("🏠 New Search"):
    st.rerun()

# Feedback Viewer
with st.expander("📂 View Submitted Feedback"):
    if os.path.exists(FEEDBACK_PATH):
        feedback_data = pd.read_csv(FEEDBACK_PATH)
        st.dataframe(feedback_data)
    else:
        st.info("No feedback submitted yet.")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>Made by Rahul 🚀 | Streamlit + ML Project</p>
""", unsafe_allow_html=True)
