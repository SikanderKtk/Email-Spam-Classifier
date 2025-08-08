import streamlit as st
import joblib
from PIL import Image

# =======================
# Page Configuration
# =======================
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# Load Model & Vectorizer
# =======================
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# =======================
# Custom CSS for Styling
# =======================
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            color: #1a73e8;
            font-weight: bold;
            padding-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #555;
            padding-bottom: 20px;
        }
        .stButton>button {
            background-color: #1a73e8;
            color: white;
            font-size: 1.1rem;
            border-radius: 8px;
            padding: 0.6rem 1.2rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #155ab6;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# =======================
# Header Section
# =======================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    email_img = Image.open("emailspam.png")  # Lowercase filename
    st.image(email_img, use_container_width=True)

st.markdown('<div class="title">📧 AI-Powered Email Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect spam emails instantly with Machine Learning</div>', unsafe_allow_html=True)

# =======================
# Main Content
# =======================
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("✉️ Enter Your Email Text")
    email_input = st.text_area("Paste email content here...", height=200)

    if st.button("🔍 Classify Email"):
        if email_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            # Transform and predict
            features = vectorizer.transform([email_input])
            prediction = model.predict(features)[0]

            if prediction == 1:
                st.error("🚨 This email is **SPAM**")
            else:
                st.success("✅ This email is **NOT SPAM**")

with col_right:
    st.subheader("📌 Email Classifier")
    poster_img = Image.open("emailspam.png")  # Consistent lowercase filename
    st.image(poster_img, caption="Email Spam Classifier Project", use_container_width=True)

# =======================
# Footer
# =======================
st.markdown("---")
st.markdown("🔹 **Developed by:** Your Name | 📅 2025 | 🚀 Powered by Naive Bayes & TF-IDF")
