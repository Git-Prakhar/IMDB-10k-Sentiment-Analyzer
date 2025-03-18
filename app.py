import streamlit as st
import pickle

with open('random_forest_model.pkl', 'rb') as file:
    rf = pickle.load(file)

with open('count_vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

def predict_sentiment(review):
    review = [review]
    review = cv.transform(review).toarray()
    return 'Positive' if rf.predict(review)[0] == 1 else 'Negative'
  

st.title("IMDB Sentiment Analysis (80% accuracy)")
review = st.text_area("Enter your review here")
if st.button("Predict"):
    sentiment = predict_sentiment(review)
    st.write(f"Sentiment: {sentiment}")