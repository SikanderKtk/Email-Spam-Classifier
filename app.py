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

# Custom CSS Styling
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #f0f0f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .container {
            background: #1a1a2e;
            padding: 2rem 2.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
            height: 500px;
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
            font-size: 1.1rem !important;
            box-shadow: inset 0 0 6px rgba(255,255,255,0.1);
            resize: vertical !important;
            min-height: 250px !important;
        }
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
            margin-top: 1rem;
        }
        div.stButton > button:hover {
            background: #e6b800;
            cursor: pointer;
            box-shadow: 0 6px 20px rgba(230, 184, 0, 0.8);
        }
        .result-box {
            background-color: #111827;
            padding: 2rem;
            border-radius: 15px;
            height: 100%;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            color: #f0f0f5;
        }
        .result-success {
            color: #16c79a;
            font-weight: 700;
            font-size: 1.5rem;
        }
        .result-error {
            color: #e74c3c;
            font-weight: 700;
            font-size: 1.5rem;
        }
        .info-box {
            background: #252a41;
            padding: 1rem;
            border-radius: 12px;
            margin-top: 1.5rem;
            font-size: 0.95rem;
            line-height: 1.4;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .examples button {
            background-color: #4b6cb7;
            border: none;
            border-radius: 8px;
            color: white;
            padding: 8px 14px;
            margin: 0 8px 8px 0;
            cursor: pointer;
            transition: background-color 0.25s ease;
        }
        .examples button:hover {
            background-color: #3a529f;
        }
        footer {
            text-align: center;
            color: #ddd;
            font-size: 0.9rem;
            margin-top: 3rem;
            letter-spacing: 0.7px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<h1 class="title">📧 AI-Powered Email Spam Classifier</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">Paste your email below and find out if it is spam or not!</h3>', unsafe_allow_html=True)

# Layout: two columns side by side
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="container">', unsafe_allow_html=True)

    email_input = st.text_area("✉️ Enter your email text here...")

    # Example buttons for easy testing
    st.markdown('<div class="examples"><strong>Try Examples:</strong></div>', unsafe_allow_html=True)
    example_spam = """Subject: Congratulations! You won a $1,000,000 prize!

Dear Winner,

We are pleased to inform you that you have been selected for a grand prize of $1,000,000 USD. To claim your reward, please provide your bank details and pay a small processing fee of $99.

Hurry! This offer expires soon.

Click the link below to claim now:
http://fakeprize-claim.com

Best regards,
Prize Department
"""
    example_ham = """Subject: Meeting Schedule Confirmation

Hi Sarah,

Just confirming our meeting tomorrow at 3 PM in the conference room. Please let me know if you need any documents prepared beforehand.

Looking forward to our discussion.

Best,
John
"""

    def set_example(text):
        st.session_state["email_input"] = text

    # Buttons in one line
    cols = st.columns(2)
    with cols[0]:
        if st.button("Load Spam Example"):
            set_example(example_spam)
    with cols[1]:
        if st.button("Load Ham Example"):
            set_example(example_ham)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    # Use session_state to keep input synced
    email_text = st.session_state.get("email_input", email_input)

    if email_text.strip() == "":
        st.info("📝 Enter an email on the left, then press **Classify Email** to see the result here.")
    else:
        if st.button("🔍 Classify Email"):
            features = vectorizer.transform([email_text])
            prediction = model.predict(features)[0]

            if prediction == 1:
                st.markdown('<p class="result-error">🚨 This email is <strong>SPAM</strong></p>', unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                    <strong>Why spam?</strong><br>
                    This email contains typical spam indicators:<br>
                    - Requests for money or personal info<br>
                    - Urgent language and suspicious links<br>
                    - Too good to be true offers
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<p class="result-success">✅ This email is <strong>NOT SPAM</strong></p>', unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                    <strong>Why not spam?</strong><br>
                    This email appears legitimate:<br>
                    - Formal greeting and sign-off<br>
                    - No suspicious requests or links<br>
                    - Clear professional language
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<footer>🔹 Developed by Muhammad Sikander Bakht | 📅 Aug 2025 | 🚀 Powered by Naive Bayes & TF-IDF</footer>', unsafe_allow_html=True)
