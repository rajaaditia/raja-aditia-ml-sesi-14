# spam.py

import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load or train model
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        # Load data
        df = pd.read_csv("SMSSpamCollection", sep="\t", header=None, names=["label", "message"])
        df['label_num'] = df.label.map({'ham':0, 'spam':1})

        # Vectorize
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['message'])
        y = df['label_num']

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, "model.pkl")
        joblib.dump(vectorizer, "vectorizer.pkl")
    else:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")

    return model, vectorizer

# Load model
model, vectorizer = load_model()

# Streamlit UI
st.title("Aplikasi Deteksi Pesan Spam")
st.markdown("Masukkan pesan teks, lalu sistem akan memprediksi apakah pesan tersebut **Spam** atau **Bukan**.")

input_text = st.text_area("Tulis pesan di sini:")

if st.button("Deteksi"):
    if input_text.strip() != "":
        input_vector = vectorizer.transform([input_text])
        prediction = model.predict(input_vector)[0]
        if prediction == 1:
            st.error("🚫 SPAM!")
        else:
            st.success("✅ Bukan Spam.")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")