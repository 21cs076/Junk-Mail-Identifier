import streamlit as st
import joblib
import os
import re

# Load your trained model
model_path = r'model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error(f"Model file not found at {model_path}")

# Load your vectorizer
vectorizer_path = r'vectorizer.pkl'
if os.path.exists(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
else:
    st.error(f"Vectorizer file not found at {vectorizer_path}")

def preprocess_text(text):
    # Remove unwanted characters and preprocess the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove all non-word characters
    text = text.strip()  # Remove leading and trailing whitespace
    return text

st.set_page_config(page_title="Spam Email Detection", page_icon="📧")

# Custom CSS to improve the UI
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 12px;
    }
    .stButton>button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('📧 Spam Email Detection')
st.write('Enter the text of the email below to check if it is spam or legitimate.')

input_text = st.text_area('Email Text', height=200)
if st.button('Predict'):
    if input_text:
        try:
            # Preprocess the input data
            data = preprocess_text(input_text)

            # Convert the input data to a format suitable for prediction
            data_vectorized = vectorizer.transform([data])

            # Make predictions using your model
            predictions = model.predict(data_vectorized)

            # Map the prediction to "Spam" or "Not Spam"
            prediction_label = "🚨 Suspected Spam" if predictions[0] == 1 else "✅ Legitimate"

            # Display the prediction
            st.markdown(f"## {prediction_label}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error('Please enter some text to predict.')
