# Email-Spam-Classifier
# 📝 Overview
Email spam remains one of the most common cybersecurity threats, with millions of spam messages sent daily.
This project aims to automatically detect spam emails using Machine Learning and Natural Language Processing (NLP) techniques.
The model classifies emails as either:

Spam – Unwanted or malicious emails.

Ham – Genuine, non-spam emails.

A clean and interactive Streamlit interface has been built to allow real-time testing of the model.

# 🔍 Methodology
1. Data Collection
Dataset containing labeled emails as Spam or Ham.

2. Data Preprocessing
Removal of punctuation, numbers, and special characters.

Conversion to lowercase.

Tokenization and stopword removal using NLTK.

Lemmatization to reduce words to their base form.

3. Feature Extraction
Applied TF-IDF Vectorization to convert email text into numerical vectors for ML models.

4. Model Training
Chose Naive Bayes as the classification algorithm due to its efficiency in text-based classification.

Split data into training and testing sets for performance evaluation.

 5. Evaluation Metrics
Accuracy

Precision

Recall

F1-Score

6. Deployment
Developed a Streamlit web app for real-time predictions.

Users can paste any email text and get a classification with a confidence score.

# 📊 Results
Algorithm Used: Naive Bayes

Accuracy Achieved: 97%

Precision: 96%

Recall: 95%

F1-Score: 95%

# 💡 Findings
TF-IDF Vectorization significantly improved accuracy compared to raw text inputs.

Naive Bayes proved to be lightweight and highly effective for spam classification.

Most false positives were due to promotional emails with mixed genuine content.

The interactive web interface improves usability for non-technical users.


# 🛠 Tech Stack
Python

Streamlit – Web interface

Scikit-learn – Model training & evaluation

NLTK – NLP preprocessing

Pandas, NumPy – Data handling


