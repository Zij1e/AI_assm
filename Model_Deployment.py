import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

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

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Buttons for actions
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Clear Canvas"):
        canvas_result.image_data.fill(0)
        st.experimental_rerun()

with col2:
    if st.button("Predict"):
        if canvas_result.image_data is not None:
            image = Image.fromarray(canvas_result.image_data.astype('uint8'))
            predicted_char, confidence = predict_character(image)
            st.write(f"The predicted character is: {predicted_char}")
            st.write(f"Confidence: {confidence:.2f}%")
        else:
            st.write("Please draw something before predicting.")

with col3:
    # Option to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        predicted_char, confidence = predict_character(image)
        st.write(f"The predicted character is: {predicted_char}")
        st.write(f"Confidence: {confidence:.2f}%")

# Instructions for users
st.markdown("""
    ### Instructions:
    1. Draw a single character on the canvas above.
    2. Click 'Predict' to see the recognition result.
    3. Use 'Clear Canvas' to start over.
    4. You can also upload an image using the 'Choose an image...' option.
""")
