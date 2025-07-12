import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('mnb.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

# App title
st.title("📩 SMS Spam Classifier")

# Text input
msg = st.text_area("Enter your message:")

# Predict button
if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess and predict
        vectorized_msg = vectorizer.transform([msg])
        result = model.predict(vectorized_msg)[0]

        if result == 1:
            st.error("🚫 Spam Message Detected!")
        else:
            st.success("✅ Not Spam")
