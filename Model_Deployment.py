#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from tensorflow.keras import utils

DATADIR = "emnist"


# In[2]:


#only 2nd column needed, has delimiters, no header

mapp = pd.read_csv(
    "C:\\Users\\chyik\\Downloads\\handwriting (2)\\handwriting\\emnist\\emnist-balanced-mapping.txt",
    delimiter=' ',
    header=None,
    usecols=[1]
)

# Convert the DataFrame to a Series
mapp = mapp.squeeze()

mapp.head(12)


# In[3]:


#testing charcodes
print("char code of index 11 is ", mapp[11])
#OR
print(chr(mapp[11]))


# In[4]:


#class mapping list to map indices w the actual char associated

class_mapping = []

for num in range(len(mapp)):
    class_mapping.append(chr(mapp[num]))


# In[5]:


#TESTING
class_mapping[46]


# # Handwriting Recognition with LSTM

# In[6]:


import tkinter as tk
from tkinter import Canvas, Button, messagebox
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

class HandwritingRecognitionApp:
    def __init__(self, master, class_mapping):
        self.master = master
        self.master.title("Handwriting Recognition with LSTM")
        
        # Load the model
        self.model = load_model('handwriting_recognition_model.h5')
        
        # Set the class mapping
        self.class_mapping = class_mapping
        
        # Create canvas
        self.canvas = Canvas(self.master, width=280, height=280, bg="black")
        self.canvas.pack(pady=20)
        
        # Create an image draw object
        self.image = Image.new("L", (280, 280), color="black")
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        # Create buttons
        self.clear_button = Button(self.master, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=10)
        
        self.predict_button = Button(self.master, text="Predict", command=self.predict_character)
        self.predict_button.pack(side=tk.RIGHT, padx=10)
        
        # Initialize drawing variables
        self.old_x = None
        self.old_y = None
    
    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                    width=20, fill="white", capstyle=tk.ROUND, 
                                    smooth=tk.TRUE, splinesteps=36)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                           fill="white", width=20, joint="curve")
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x = None
        self.old_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
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
        
        # Show result
        messagebox.showinfo("Prediction", f"The predicted character is: {predicted_char}\nConfidence: {confidence:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Use your existing class_mapping here
    app = HandwritingRecognitionApp(root, class_mapping)
    root.mainloop()


# In[ ]:




