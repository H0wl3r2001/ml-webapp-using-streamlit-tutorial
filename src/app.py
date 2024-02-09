import streamlit as st
from pickle import load
import regex as re

# Load models and vectorizer
vector = load(open("models/tfidf.sav", "rb"))
model = load(open("models/SVM.sav", "rb"))

def preprocess_text(text):
    # Apply the preprocessing steps directly to the input text
    text = re.sub(r'[^a-z ]', " ", text)
    text = re.sub(r'\s+[a-zA-Z]\s+', " ", text)
    text = re.sub(r'\^[a-zA-Z]\s+', " ", text)
    text = re.sub(r'\s+', " ", text.lower())
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
    
    return text



def predict_sentiment(str_):
    sentence_vector = vector.transform([preprocess_text(str_)]).toarray()
    prediction = model.predict(sentence_vector)
    if prediction == 1:
        return 'SPAM'
    else:
        return 'Not a spam email'


# Define Streamlit app
def main():
    st.title('Email Spam Detector')
    st.write('Enter your email text, in the format "Subject: [text]":')
    
    # Input text box for user to enter email text
    user_input = st.text_input('Enter email text here:')
    
    # Button to trigger prediction
    if st.button('Detect'):
        if user_input.strip() == '':
            st.write('Enter a sample email text')
        else:
            result = predict_sentiment(user_input)
            st.write('Sentiment:', result)

if __name__ == '__main__':
    main()
