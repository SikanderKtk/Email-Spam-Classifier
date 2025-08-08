import streamlit as st
import pickle
import re
import string

# Load model & vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Text cleaning function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Page Config
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="wide")

# CSS Styling
st.markdown("""
    <style>
        .main-card {
            background-color: #f9fafb;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin-top: 2rem;
        }
        h1 {
            text-align: center;
            color: #2C3E50;
        }
        .stButton>button {
            background-color: #3498DB;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 5px;
            border: none;
            font-size: 1rem;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #2980B9;
        }
        .result-box {
            padding: 1rem;
            border-radius: 8px;
            font-size: 1.1rem;
            margin-top: 1rem;
            text-align: center;
            font-weight: bold;
        }
        .spam {
            background-color: #FDEDEC;
            color: #C0392B;
        }
        .ham {
            background-color: #E8F8F5;
            color: #148F77;
        }
    </style>
""", unsafe_allow_html=True)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.title("📧 Email Spam Classification")
    st.markdown("Detect whether an email is **Spam** or **Ham (Not Spam)** instantly.")

    # Input
    user_input = st.text_area("✉️ Paste your email content here:", height=150, placeholder="Type or paste the email text...")

    # Button
    if st.button("🔍 Classify Email"):
        if user_input.strip() != "":
            processed_text = preprocess_text(user_input)
            input_vector = vectorizer.transform([processed_text])

            prediction = model.predict(input_vector)[0]
            prediction_proba = model.predict_proba(input_vector)[0]

            if prediction == 1:
                st.markdown(
                    f"<div class='result-box spam'>🚨 This email is **Spam**! <br> Confidence: {prediction_proba[1]*100:.2f}%</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='result-box ham'>✅ This email is **Ham (Not Spam)**! <br> Confidence: {prediction_proba[0]*100:.2f}%</div>",
                    unsafe_allow_html=True
                )
        else:
            st.warning("⚠️ Please enter some text before classifying.")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Spam_email.png/640px-Spam_email.png",
        caption="Sample Spam Email",
        use_container_width=True
    )
