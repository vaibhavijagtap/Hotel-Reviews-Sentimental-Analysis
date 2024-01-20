import streamlit as st
import pickle
import numpy as np


# Load model
model = pickle.load(open('trained_model.sav', 'rb'))





# Create title
st.title('Hotel Review Sentiment')
review = st.text_input('Enter your review below')
submit = st.button('Predict')





if submit:
    prediction = model.predict([review])[0]

    # Emoji representation based on sentiment
    if prediction == 'Positive':
        st.success('Positive Review')
        emoji = '😃'
    elif prediction == 'Negative':
        st.warning('Negative Review')
        emoji = '😞'
    else:
        st.success('Neutral Review')
        emoji = '😐'

    st.write(f'{emoji}')


# Slider
rating = st.slider('Please rate your overall satisfication with the hotel', min_value=1, max_value=5, step=1)





