# [Junk Mail Identifier](https://spam-email-detection-wtwvi5hz465ezjbcmwj7j8.streamlit.app/)

## Introduction
This is focused on detecting spam emails using machine learning techniques. Here is a detailed explanation of its contents and purpose:

1. **Repository Overview**:
    - **Description**: The repository aims to detect spam emails.
    - **Languages Used**:
        - Python: 100%

2. **Key Files**:
    - **train.ipynb**: A Jupyter Notebook used to train different machine learning models for spam detection.

3. **Training Process in `train.ipynb`**:
    - **Libraries Imported**:
        - `pandas` for data manipulation.
        - `scikit-learn` for machine learning tasks.
        - `joblib` for saving models.
    - **Data Loading**:
        - The dataset is loaded from a CSV file.
    - **Data Preprocessing**:
        - Unnecessary columns are dropped, and remaining columns are renamed.
        - The text data is transformed into numerical features using `TfidfVectorizer`.
        - Labels are mapped from text categories ('ham' and 'spam') to numerical values.
    - **Model Training**:
        - Several models are trained, including:
            - Support Vector Machine (SVM) with linear and RBF kernels.
            - Naive Bayes.
            - Logistic Regression.
    - **Model Evaluation**:
        - The models are evaluated using metrics like precision, recall, and F1-score.
    - **Model Saving**:
        - The best-performing model and vectorizer are saved as `.pkl` files.

The repository provides a comprehensive approach to building and evaluating models for spam email detection.

## Relevance

The `Junk Mail Identifier` project is relevant for several reasons:

- **Email Security**: It helps in identifying and filtering out spam emails, which can contain malicious links or attachments that pose security threats.
- **Productivity**: By filtering spam, it reduces clutter in email inboxes, allowing users to focus on important emails.
- **Machine Learning Application**: Demonstrates the application of machine learning techniques in real-world problems, specifically in text classification.
- **Educational Value**: Provides a learning resource for understanding how to preprocess text data, train machine learning models, and evaluate their performance.

## Dataset

We have used (a public dataset): [Spam detection using Scikit learn](https://www.kaggle.com/code/yakinrubaiat/spam-detection-using-scikit-learn).

The dataset contains **5,728 entries** with the following two columns:

1. **`text`**: A string column containing email content, including the subject line and the body of the email.
2. **`spam`**: An integer column (binary) indicating whether the email is spam (1) or not spam (0).

### Key Details:
- **Data type**:
  - `text`: Object (string).
  - `spam`: Integer.
- There are no missing values in either column.

## Model Training

This is used to train multiple machine learning models to detect spam emails. Here’s a detailed explanation of the notebook:

1. **Import Libraries**:
    ```python
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    ```
    - Various libraries are imported, including `pandas` for data manipulation, `scikit-learn` for machine learning tasks, and `joblib` for saving models.

2. **Load Dataset**:
    ```python
    data = pd.read_csv(r'H:\\Project\\Spam Email Detection\\emails.csv', encoding='latin-1')
    ```
    - The dataset is loaded from a CSV file. The encoding is specified as 'latin-1'.

3. **Data Preprocessing**:
    ```python
    data = data[['text', 'spam']]
    data.columns = ['EmailText', 'Label']
    ```
    - Unnecessary columns are dropped, and remaining columns are renamed for clarity.

4. **Vectorization and Saving Vectorizer**:
    ```python
    import joblib
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['EmailText'])
    y = data['Label'].map({'ham': 0, 'spam': 1})
    joblib.dump(vectorizer, 'vectorizer.pkl')
    ```
    - The email text is transformed into TF-IDF features, and labels are mapped to numerical values (ham = 0, spam = 1).
    - The vectorizer is saved to a file.

5. **Handle Missing Values and Split Data**:
    ```python
    y = data['Label'].dropna()
    X = X[data['Label'].notna()]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```
    - Missing values in the target variable are handled.
    - The dataset is split into training and testing sets.

6. **Model Training and Saving**:
    - **Support Vector Machine (Linear Kernel)**:
        ```python
        svm_linear = SVC(kernel='linear')
        svm_linear.fit(X_train, y_train)
        y_pred_svm_linear = svm_linear.predict(X_test)
        joblib.dump(svm_linear, 'model.pkl')
        ```
        - Trains an SVM with a linear kernel and saves the model.
    
    - **Support Vector Machine (RBF Kernel)**:
        ```python
        svm_rbf = SVC(kernel='rbf')
        svm_rbf.fit(X_train, y_train)
        y_pred_svm_rbf = svm_rbf.predict(X_test)
        ```
        - Trains an SVM with an RBF kernel.

    - **Naive Bayes**:
        ```python
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)
        ```
        - Trains a Naive Bayes classifier.

    - **Logistic Regression**:
        ```python
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        ```
        - Trains a Logistic Regression model.

7. **Model Evaluation**:
    - **SVM (Linear Kernel)**:
        ```python
        print("SVM (Linear Kernel) Classification Report:")
        print(classification_report(y_test, y_pred_svm_linear))
        ```
        - Prints the classification report for SVM with a linear kernel.

    - **SVM (RBF Kernel)**:
        ```python
        print("SVM (RBF Kernel) Classification Report:")
        print(classification_report(y_test, y_pred_svm_rbf))
        ```
        - Prints the classification report for SVM with an RBF kernel.

    - **Naive Bayes**:
        ```python
        print("Naive Bayes Classification Report:")
        print(classification_report(y_test, y_pred_nb))
        ```
        - Prints the classification report for Naive Bayes.

    - **Logistic Regression**:
        ```python
        print("Logistic Regression Classification Report:")
        print(classification_report(y_test, y_pred_lr))
        ```
        - Prints the classification report for Logistic Regression.

The script loads and preprocesses the dataset, trains several models, evaluates their performance, and saves the best-performing model and vectorizer for future use.

## Model

This is likely a serialized machine learning model used for detecting spam emails. Here's a detailed explanation of the model's structure and purpose based on typical usage:

1. **Purpose**:
    - The model is designed to classify emails as either "Spam" or "Not Spam" based on the content of the email text.

2. **Structure**:
    - **Vectorizer**: Before training the model, the text data is transformed into numerical features using a vectorizer. In this case, a `TfidfVectorizer` is commonly used to convert text into TF-IDF features, which capture the importance of words in the context of the emails.
    - **Model**: The trained model could be one of several types, such as:
        - **Support Vector Machine (SVM)**: A powerful classifier that finds the optimal hyperplane to separate different classes.
        - **Naive Bayes**: A probabilistic classifier based on Bayes' theorem, often used for text classification.
        - **Logistic Regression**: A linear model used for binary classification tasks.
    - **Label Mapping**: The target labels are typically mapped from text categories (e.g., 'ham' for non-spam and 'spam' for spam) to numerical values (e.g., 0 for non-spam and 1 for spam).

3. **Training Process**:
    - **Data Loading and Preprocessing**: The dataset is loaded, unnecessary columns are dropped, and the text is preprocessed (e.g., removing special characters, converting to lowercase).
    - **Feature Extraction**: The email text is transformed into numerical features using the vectorizer.
    - **Model Training**: The model is trained on the processed features and corresponding labels.
    - **Model Evaluation**: The performance of the model is evaluated using metrics like precision, recall, and F1-score.
    - **Model Saving**: The trained model and vectorizer are saved to `.pkl` files for future use.

4. **Usage**:
    - **Loading the Model**: The saved model and vectorizer are loaded using a tool like `joblib`.
    - **Prediction**: New email text is preprocessed and transformed using the vectorizer, and then the model predicts whether the email is spam or not.

This model is essential for automatically identifying spam emails, helping to filter them out and reduce unwanted messages in email systems.

## Vectorizer

This is a serialized vectorizer that is used to transform text data into numerical features. Here's a detailed explanation of the vectorizer's structure and purpose:

1. **Purpose**:
    - The vectorizer is used to convert the raw email text into a format that can be used by the machine learning model to make predictions. Specifically, it transforms the text into TF-IDF (Term Frequency-Inverse Document Frequency) features.

2. **Structure**:
    - **TF-IDF Vectorizer**: The `vectorizer.pkl` file contains a `TfidfVectorizer` object from the `scikit-learn` library. This object is trained on the email dataset and captures the importance of each word in the context of the emails.
        - **Vocabulary**: The vectorizer has a vocabulary that maps each word to an index.
        - **IDF Values**: The vectorizer stores the inverse document frequency values for each word, which helps in scaling the term frequency by how rare the word is across all documents.

3. **Training Process**:
    - **Data Loading**: The email text dataset is loaded.
    - **Text Preprocessing**: The text is cleaned by removing special characters, converting to lowercase, and removing stop words.
    - **Vectorizer Fitting**: The cleaned text is used to fit the `TfidfVectorizer`, which learns the vocabulary and IDF values from the data.
    - **Vectorizer Saving**: The fitted vectorizer is saved to a file (`vectorizer.pkl`) using `joblib`.

4. **Usage**:
    - **Loading the Vectorizer**: The saved vectorizer is loaded using `joblib`.
    - **Transforming Text**: New email text is preprocessed and transformed into TF-IDF features using the loaded vectorizer.
    - **Model Prediction**: The transformed text features are fed into the machine learning model to predict whether the email is spam or not.

The vectorizer is essential for converting text data into numerical features that the machine learning model can understand and use to make accurate predictions.

## Deploy

This is designed to create a web application using Streamlit to detect spam emails. Here’s a detailed explanation of the script:

1. **Import Libraries**:
    - `streamlit as st`: Streamlit is used to build the web application.
    - `joblib`: Used to load the pre-trained model and vectorizer.
    - `os`: Used to check if the model and vectorizer files exist.
    - `re`: Used for text preprocessing.

2. **Load the Trained Model**:
    ```python
    model_path = r'model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.error(f"Model file not found at {model_path}")
    ```
    - The script checks if the `model.pkl` file exists. If it does, it loads the model using `joblib`. If not, it displays an error message.

3. **Load the Vectorizer**:
    ```python
    vectorizer_path = r'vectorizer.pkl'
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        st.error(f"Vectorizer file not found at {vectorizer_path}")
    ```
    - Similarly, it checks for the `vectorizer.pkl` file and loads it if available; otherwise, it shows an error.

4. **Text Preprocessing Function**:
    ```python
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
        text = re.sub(r'[^\w\s]', '', text)  # Remove all non-word characters
        text = text.strip()  # Remove leading and trailing whitespace
        return text
    ```
    - This function cleans the input text by:
      - Replacing multiple whitespaces with a single space.
      - Removing non-word characters (punctuation, etc.).
      - Stripping leading and trailing whitespace.

5. **Streamlit Application**:
    ```python
    st.title('Spam Email Detection')
    st.write('Enter the text of the email below:')
    ```
    - Sets the title of the web application and provides instructions to the user.

6. **Input Text and Prediction**:
    ```python
    input_text = st.text_area('Email Text')
    if st.button('Predict'):
        if input_text:
            try:
                data = preprocess_text(input_text)
                data_vectorized = vectorizer.transform([data])
                predictions = model.predict(data_vectorized)
                prediction_label = "Suspected Spam" if predictions[0] == 1 else "Legitimate"
                st.markdown(f"{prediction_label}")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error('Please enter some text to predict.')
    ```
    - Creates a text area for the user to input email text.
    - When the "Predict" button is clicked:
      - Checks if the input text is provided.
      - Preprocesses the input text.
      - Transforms the text into a format suitable for the model using the vectorizer.
      - Uses the model to predict whether the email is spam or legitimate.
      - Displays the result as "Suspected Spam" or "Legitimate".
      - If an error occurs during prediction, it shows an error message.
      - If no text is entered, it prompts the user to enter some text.

The script essentially sets up a simple web interface to detect spam emails using a pre-trained machine learning model.
