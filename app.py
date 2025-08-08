import streamlit as st
import joblib

# =======================
# Page Configuration
# =======================
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =======================
# Load Model & Vectorizer
# =======================
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# =======================
# Custom CSS Styling
# =======================
st.markdown("""
    <style>
        /* Background and font */
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #f0f0f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Main container style */
        .main-container {
            background: #1a1a2e;
            padding: 2rem 3rem;
            border-radius: 15px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
            max-width: 700px;
            margin: 3rem auto 5rem auto;
        }
        /* Title style */
        .title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            color: #ffcc00;
            margin-bottom: 0.2rem;
            letter-spacing: 1.5px;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        }
        /* Subtitle style */
        .subtitle {
            text-align: center;
            font-size: 1.25rem;
            color: #dcdcdc;
            margin-bottom: 2.5rem;
            font-weight: 500;
        }
        /* Text area style */
        textarea {
            background: #0f3460 !important;
            color: #f5f5f7 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            font-size: 1.1rem !important;
            box-shadow: inset 0 0 6px rgba(255,255,255,0.1);
            resize: vertical !important;
            min-height: 180px !important;
        }
        /* Button style */
        div.stButton > button {
            background: #ffcc00;
            color: #1a1a2e;
            font-weight: 700;
            font-size: 1.2rem;
            padding: 0.7rem 1.6rem;
            border-radius: 12px;
            border: none;
            width: 100%;
            transition: background 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 204, 0, 0.5);
        }
        div.stButton > button:hover {
            background: #e6b800;
            cursor: pointer;
            box-shadow: 0 6px 20px rgba(230, 184, 0, 0.8);
        }
        /* Result message styling */
        .stSuccess, .stError, .stWarning {
            border-radius: 12px;
            padding: 1rem 1.2rem;
            font-size: 1.15rem;
            margin-top: 1.5rem;
            font-weight: 600;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            box-shadow: 0 4px 14px rgba(0,0,0,0.3);
        }
        .stSuccess {
            background-color: #16c79a;
            color: white;
        }
        .stError {
            background-color: #e74c3c;
            color: white;
        }
        .stWarning {
            background-color: #f39c12;
            color: white;
        }
        /* Footer style */
        footer {
            text-align: center;
            color: #ddd;
            font-size: 0.9rem;
            margin-top: 4rem;
            letter-spacing: 0.7px;
        }
    </style>
""", unsafe_allow_html=True)

# =======================
# App content container
# =======================
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown('<h1 class="title">📧 AI-Powered Email Spam Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect spam emails instantly with Machine Learning</p>', unsafe_allow_html=True)

email_input = st.text_area("✉️ Enter your email text here...")

if st.button("🔍 Classify Email"):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        features = vectorizer.transform([email_input])
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.error("🚨 This email is **SPAM**")
        else:
            st.success("✅ This email is **HAM (NOT SPAM)**")

st.markdown("</div>", unsafe_allow_html=True)

# =======================
# Footer
# =======================
st.markdown('<footer>🔹 Developed by Sikander Bakht | 📅 Aug 2025 | 🚀 Powered by Naive Bayes & TF-IDF</footer>', unsafe_allow_html=True)
