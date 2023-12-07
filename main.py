import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from io import BytesIO

# Function to load the combined model
@st.cache(allow_output_mutation=True)
def load_model():
    # URLs for model parts on GitHub
    base_url = "https://github.com/m3mentomor1/Breast-Cancer-Image-Classification/raw/main/"
    model_parts = [f"{base_url}best_model.zip.{i:03d}" for i in range(1, 40)]

    # Download and combine model parts
    model_bytes = b''
    for part_url in model_parts:
        response = requests.get(part_url)
        model_bytes += response.content

    # Load the combined model
    model = tf.keras.models.load_model(BytesIO(model_bytes))
    return model

# Load the model
model = load_model()

# Function to preprocess and make predictions
def predict(image):
    # Preprocess the image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, (256, 256))  # Adjust the size as per your model requirements
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make prediction
    predictions = model.predict(img_array)
    return predictions

# Streamlit app
st.title('Breast Cancer Image Classification')
uploaded_file = st.file_uploader("Choose a breast ultrasound image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Make predictions
    predictions = predict(image)
    st.write(f"Predictions: {predictions}")
