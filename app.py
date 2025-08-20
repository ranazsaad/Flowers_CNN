import streamlit as st
import tensorflow as tf
from keras.utils import img_to_array  
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("cnn_model.keras")

# Flower class names
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Streamlit page config
st.set_page_config(
    page_title="🌸 Flower Classifier",
    page_icon="🌼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark lavender theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #4b2142;  /* dark lavender */
        color: #e6d6f2;  /* soft lavender text */
    }
    h1, h2, h3, h4, h5, h6, p, label {
        color: #e6d6f2 !important;
    }
    .stFileUploader label {
        color: #e6d6f2 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #9d4edd, #7b2cbf);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.5em 1.5em;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #7b2cbf, #5a189a);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='text-align: center;'>🌸 Flower Classifier 🌸</h1>", unsafe_allow_html=True)
st.write("Upload an image of a flower, and the model will predict its type.")

# File uploader
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((128, 128))
    st.image(img, caption="🌺 Uploaded Image 🌺", use_column_width=True)

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(
        f"<h2>✨ This flower is a <b>{predicted_class}</b> 🌼 ",
        unsafe_allow_html=True
    )
