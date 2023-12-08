import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import h5py

# Function to load the combined model
@st.cache(allow_output_mutation=True)
def load_model():
    # URLs for model parts on GitHub
    base_url = "https://github.com/m3mentomor1/Breast-Cancer-Image-Classification/raw/main/"
    model_parts = [f"{base_url}best_model.hdf5.h5.part{i:02d}" for i in range(1, 27)]

    # Download and combine model parts
    model_bytes = b''
    for part_url in model_parts:
        response = requests.get(part_url)
        model_bytes += response.content

    # Create an in-memory HDF5 file
    with h5py.File(BytesIO(model_bytes), 'r') as hf:
        # Load the combined model
        model = tf.keras.models.load_model(hf)

    return model

# Function to preprocess and make predictions
def predict(image, model):
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

    # Load the model
    model = load_model()

    # Make predictions
    predictions = predict(image, model)
    st.write(f"Predictions: {predictions}")
