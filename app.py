import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model & vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# CSS Styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #f0f0f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        /* Remove fixed height to avoid big empty box */
        .container {
            background: #1a1a2e;
            padding: 2rem 2.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
            /* Remove fixed height */
            min-height: 320px;
            max-height: 80vh;
            overflow-y: auto;
        }
        h1.title {
            font-size: 3rem;
            font-weight: 800;
            color: #ffcc00;
            margin-bottom: 0.3rem;
            letter-spacing: 1.5px;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
            text-align: center;
        }
        h3.subtitle {
            color: #dcdcdc;
            font-weight: 500;
            margin-bottom: 2rem;
            text-align: center;
        }
        textarea {
            background: #0f3460 !important;
            color: #f5f5f7 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            font-size: 1.15rem !important;
            box-shadow: inset 0 0 6px rgba(255,255,255,0.1);
            resize: vertical !important;
            min-height: 180px !important;
            max-height: 60vh !important;
            line-height: 1.5;
            overflow-y: auto !important;
        }
        div.stButton > button {
            background: #ffcc00;
            color: #1a1a2e;
            font-weight: 700;
            font-size: 1.3rem;
            padding: 0.75rem 1.8rem;
            border-radius: 12px;
            border: none;
            width: 100%;
            transition: background 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 204, 0, 0.5);
            margin-top: 1.5rem;
        }
        div.stButton > button:hover {
            background: #e6b800;
            cursor: pointer;
            box-shadow: 0 6px 20px rgba(230, 184, 0, 0.8);
        }
        .result-box {
            background-color: #111827;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 12px 24px rgba(0,0,0,0.6);
            color: #f0f0f5;
            max-height: 60vh;
            overflow-y: auto;
        }
        .result-success {
            color: #16c79a;
            font-weight: 800;
            font-size: 1.6rem;
            margin-bottom: 0.8rem;
        }
        .result-error {
            color: #e74c3c;
            font-weight: 800;
            font-size: 1.6rem;
            margin-bottom: 0.8rem;
        }
        .info-box {
            background: #252a41;
            padding: 1rem 1rem;
            border-radius: 12px;
            margin-top: 1.5rem;
            font-size: 1rem;
            line-height: 1;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            white-space: pre-line;
            color: #ddd;  /* make sure text is visible */
        }
        footer {
            text-align: center;
            color: #111827;
            font-size: 0.9rem;
            margin-top: 2rem;
            letter-spacing: 0.6px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<h1 class="title">📧 AI-Powered Email Spam Classifier</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">Paste your email below and find out if it is spam or not!</h3>', unsafe_allow_html=True)

# Layout: two columns side by side
col1, col2 = st.columns([1, 1])

with col1:

    email_input = st.text_area("✉️ Enter your email text here...")

    classify_clicked = st.button("🔍 Classify Email")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    if not classify_clicked or email_input.strip() == "":
        st.info("📝 Enter an email on the left and click **Classify Email** to see the result here.")
    else:
        features = vectorizer.transform([email_input])
        prediction = model.predict(features)[0]
   

        if prediction == 1:
            st.markdown('<p class="result-error">🚨 This email is <strong>SPAM</strong></p>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
<strong>Why is this email classified as SPAM?</strong>

- Contains urgent or threatening language (e.g., “Hurry!”, “Expires soon”).
- Requests for money, payments, or personal info.
- Suspicious or unfamiliar links.
- Too-good-to-be-true offers or prizes.
Be cautious with such emails and never share sensitive info or click suspicious links.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<p class="result-success">✅ This email is <strong>NOT SPAM</strong></p>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
<strong>Why is this email classified as NOT SPAM?</strong>

- Polite, formal language and clear purpose.
- No requests for money or sensitive information.
- No suspicious links or attachments.
- Proper greetings and sign-offs.

Always remain vigilant but this email appears legitimate.
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<footer>🔹 Developed by Muhammad Sikander Bakht | 📅 Aug 2025 | 🚀 Powered by Naive Bayes & TF-IDF</footer>', unsafe_allow_html=True)






