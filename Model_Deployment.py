import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Initialize session state
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

# Load the model
@st.cache_resource
def load_keras_model():
    return load_model('handwriting_recognition_model.h5')

model = load_keras_model()

# Load class mapping
@st.cache_data
def load_class_mapping():
    mapp = pd.read_csv(
        "emnist-balanced-mapping.txt",
        delimiter=' ',
        header=None,
        usecols=[1]
    )
    mapp = mapp.squeeze()
    return [chr(mapp[i]) for i in range(len(mapp))]

class_mapping = load_class_mapping()

def predict_character(image):
    # Preprocess the image
    image = image.resize((28, 28)).convert('L')
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0
    
    # Make prediction
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    
    # Get the predicted character
    predicted_char = class_mapping[class_index]
    
    # Calculate confidence
    confidence = np.max(prediction) * 100
    
    return predicted_char, confidence

# Streamlit UI
st.title("Handwriting Recognition with LSTM")

# Instructions for users
st.markdown("""
    ### Instructions:
    1. Draw a single character on the canvas below.
    2. Click 'Predict' to see the recognition result.
    3. Use 'Clear Canvas' to start over.
""")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
)
