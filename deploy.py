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

st.title('Spam Email Detection')
st.write('Enter the text of the email below:')

input_text = st.text_area('Email Text')
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
            prediction_label = "Suspected Spam" if predictions[0] == 1 else "Legitimate"

            # Display the prediction
            st.markdown(f"{prediction_label}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error('Please enter some text to predict.')
