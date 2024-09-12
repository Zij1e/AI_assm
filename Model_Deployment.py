import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

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
    image = image.resize((28, 28))
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

# Create a canvas for drawing
canvas_result = st.empty()
canvas = st.empty()

# Create a PIL Image for drawing
image = Image.new("L", (280, 280), color="black")
draw = ImageDraw.Draw(image)

# Function to handle mouse events for drawing
def on_mouse_move(event):
    x, y = event.x, event.y
    if event.buttons:
        draw.line([x, y, x+1, y+1], fill="white", width=20)
        canvas.image(image)

# Create a canvas using Streamlit's custom component
canvas_result = st.empty()
canvas = st.empty()

# Buttons for actions
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Clear Canvas"):
        image = Image.new("L", (280, 280), color="black")
        draw = ImageDraw.Draw(image)
        canvas.image(image)

with col2:
    if st.button("Predict"):
        predicted_char, confidence = predict_character(image)
        st.write(f"The predicted character is: {predicted_char}")
        st.write(f"Confidence: {confidence:.2f}%")

with col3:
    # Option to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        canvas.image(image)

# Display the canvas
canvas.image(image)

# Instructions for users
st.markdown("""
    ### Instructions:
    1. Draw a single character on the canvas above.
    2. Click 'Predict' to see the recognition result.
    3. Use 'Clear Canvas' to start over.
    4. You can also upload an image using the 'Choose an image...' option.
""")
