import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import io

class HandwritingRecognitionApp:
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping
        self.model = load_model('handwriting_recognition_model.h5')
        self.image = Image.new("L", (280, 280), color="black")
        self.draw = ImageDraw.Draw(self.image)

    def run(self):
        st.title("Handwriting Recognition with LSTM")

        # Create a canvas using Streamlit's drawing tool
        canvas_result = st.empty()
        stroke_width = st.slider("Stroke width:", 1, 25, 20)
        
        # Create a canvas
        canvas_result = st.empty()
        canvas_image = self.image.copy()
        canvas_draw = ImageDraw.Draw(canvas_image)

        # Handle mouse events for drawing
        canvas_placeholder = st.empty()
        canvas_placeholder.image(canvas_image, caption='Draw here', use_column_width=True)

        # Buttons
        col1, col2 = st.columns(2)
        if col1.button("Clear"):
            self.clear_canvas()
            canvas_placeholder.image(self.image, caption='Draw here', use_column_width=True)
        
        if col2.button("Predict"):
            prediction = self.predict_character()
            st.write(f"The predicted character is: {prediction['char']}")
            st.write(f"Confidence: {prediction['confidence']:.2f}%")

    def clear_canvas(self):
        self.image = Image.new("L", (280, 280), color="black")
        self.draw = ImageDraw.Draw(self.image)

    def predict_character(self):
        # Preprocess the image
        image = self.image.resize((28, 28))
        image = np.array(image)
        image = image.reshape(1, 28, 28, 1)
        image = image / 255.0
        
        # Make prediction
        prediction = self.model.predict(image)
        class_index = np.argmax(prediction)
        
        # Get the predicted character
        predicted_char = self.class_mapping[class_index]
        
        # Calculate confidence
        confidence = np.max(prediction) * 100
        
        return {"char": predicted_char, "confidence": confidence}

if __name__ == "__main__":
    # Use your existing class_mapping here
    class_mapping = [chr(i) for i in range(65, 91)]  # Example: A-Z
    app = HandwritingRecognitionApp(class_mapping)
    app.run()
