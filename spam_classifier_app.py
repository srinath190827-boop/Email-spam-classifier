import streamlit as st
import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Spam Zero", layout="centered")

# -----------------------------
# Custom CSS (for design like your image)
# -----------------------------
st.markdown("""
    <style>
    body {
        background-color: #0b0f2f;
        color: white;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #ff4b8b;
    }
    .subtitle {
        text-align: center;
        color: #bbb;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    .ham {
        background-color: #1abc9c;
        color: white;
    }
    .spam {
        background-color: #e74c3c;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title Section
# -----------------------------
st.markdown('<div class="title">SPAM ZERO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Premium Email & SMS Classifier</div>', unsafe_allow_html=True)

st.write("")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df

df = load_data()

# -----------------------------
# Preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

df['message'] = df['message'].apply(clean_text)

# -----------------------------
# Vectorization + Model
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# -----------------------------
# Input Box
# -----------------------------
message = st.text_area("Enter your message")

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("PREDICT"):
    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        cleaned = clean_text(message)
        vector = vectorizer.transform([cleaned])
        
        prediction = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]

        ham_score = prob[0] * 100
        spam_score = prob[1] * 100

        # -----------------------------
        # Output UI
        # -----------------------------
        if prediction == 0:
            st.markdown(
                f'<div class="result-box ham">✅ PREDICTION: HAM<br>This is a safe, normal message.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box spam">⚠️ PREDICTION: SPAM<br>This message looks dangerous.</div>',
                unsafe_allow_html=True
            )

        st.write("")
        st.subheader("Confidence Scores")
        st.write(f"Safe (Ham): {ham_score:.2f}%")
        st.write(f"Spam (Danger): {spam_score:.2f}%")