import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Set page config
st.set_page_config(page_title="Handwriting Recognition", page_icon="✍️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .result-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

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

# Sidebar
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Draw a single character on the canvas.
2. Click 'Predict' to see the recognition result.
3. Use 'Clear Canvas' to start over.
""")

# Main content
st.title("✍️ Handwriting Recognition with LSTM")

col1, col2 = st.columns([2, 1])

with col1:
    # Create a canvas for drawing
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="#f0f0f0",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )

with col2:
    st.markdown("<br>" * 4, unsafe_allow_html=True)  # Add some vertical space
    if st.button("Predict", key="predict_button"):
        if canvas_result.image_data is not None:
            image = Image.fromarray(canvas_result.image_data.astype('uint8'))
            predicted_char, confidence = predict_character(image)
            
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown("<p class='big-font'>Prediction Result:</p>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; font-size: 72px;'>{predicted_char}</h1>", unsafe_allow_html=True)
            st.progress(confidence / 100)
            st.write(f"Confidence: {confidence:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Please draw something before predicting.")
    
    if st.button("Clear Canvas"):
        st.session_state.canvas_key += 1
        st.experimental_rerun()
