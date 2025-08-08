import streamlit as st
import pickle
import joblib
import numpy as np
from PIL import Image

# --- Load model and vectorizer ---
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Page config ---
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="wide")

# --- Custom CSS for styling ---
st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
            padding: 2rem;
            border-radius: 10px;
        }
        .title {
            font-size: 2.2rem;
            font-weight: bold;
            color: #2C3E50;
        }
        .sub {
            font-size: 1rem;
            color: #555;
        }
        .result {
            padding: 1rem;
            border-radius: 8px;
            font-size: 1.3rem;
            font-weight: bold;
            text-align: center;
        }
        .spam {
            background-color: #ffcccc;
            color: #a10000;
        }
        .ham {
            background-color: #ccffcc;
            color: #006600;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("<p class='title'>📧 Email Spam Classification App</p>", unsafe_allow_html=True)
st.markdown("<p class='sub'>Detect whether an email is Spam or Not Spam using a trained Naive Bayes model.</p>", unsafe_allow_html=True)

st.write("---")

# --- Layout with 2 columns ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("✏️ Enter Email Content")
    user_input = st.text_area("Paste the email text here...", height=200)

    if st.button("🔍 Classify Email"):
        if user_input.strip() != "":
            # Vectorize and predict
            input_data = vectorizer.transform([user_input])
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.markdown("<div class='result spam'>🚫 This email is SPAM</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result ham'>✅ This email is NOT SPAM</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text before classifying.")

with col2:
    st.subheader("📌 Sample Email")
    sample_img = Image.open("sample_email.png")  # <-- Add a nice image in repo
    st.image(sample_img, caption="Example Email Format", use_container_width=True)

# --- Footer ---
st.write("---")
st.markdown("**Developed by Muhammad Sikander Bakht** | Data Science Student | UET Peshawar")
