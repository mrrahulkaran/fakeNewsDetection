import pandas as pd
import numpy as np
import re
import string
import nltk
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Ensure necessary resources are available
nltk.download('stopwords')

# Load Fake and True datasets
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

# Add labels
fake_df['label'] = 1  # 1 = Fake
true_df['label'] = 0  # 0 = Real

# Merge and shuffle
data = pd.concat([fake_df, true_df], ignore_index=True)
data = data[['title', 'text', 'label']].dropna()
data['content'] = data['title'] + " " + data['text']

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in words if w not in stop_words]
    return ' '.join(filtered)

# Apply cleaning
data['cleaned'] = data['content'].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['cleaned']).toarray()
y = data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/fake_news_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("✅ Model and vectorizer saved in 'models/' folder.")
