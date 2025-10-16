import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load trained model
try:
    model = tf.keras.models.load_model("alphabet_recognition_model.h5")
    classes = [chr(i) for i in range(65, 91)]  # A-Z
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

st.set_page_config(page_title="✍️ Handwritten Character Recognition", layout="centered")

# CSS Styling
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        color: #4c9a2a; /* Green */
        font-weight: bold;
        margin-bottom: 30px;
        text-shadow: 2px 2px 4px #00000020; /* Subtle shadow */
    }
    .subheader {
        text-align: center;
        font-size: 18px;
        color: #757575;
    }
    .stButton > button:first-child {
        padding: 10px 25px;
        color: white;
        background-color: #4c9a2a; /* Green */
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: background-color 0.3s ease;
        box-shadow: 2px 2px 5px #00000010;
    }
    .stButton > button:first-child:hover {
        background-color: #00441b;
    }
    .prediction {
        font-size: 36px;
        color: #4c9a2a; /* Green */
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
        padding: 15px;
        background-color: #acdf87; /* Light green */
        border-radius: 10px;
        box-shadow: 2px 2px 8px #00000010;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown('<div class="title">✍️ Handwritten Character Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Draw a single uppercase letter [A-Z] on the canvas and let the AI predict it in real-time!</div>', unsafe_allow_html=True)
st.markdown("---")

# Canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=12,
    stroke_color="#212121",  # Dark grey
    background_color="#f5f5f5",  # Light grey
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction Logic
if st.button("🔍 Predict"):
    if canvas_result.image_data is not None:
        # Convert RGBA to grayscale
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        gray_array = np.array(img)
        
        # Making sure the canvas is not empty
        if np.all(gray_array == 245):
            st.warning("The canvas is blank. Please draw a character.")
        else:
            # Resize to 28x28
            img = ImageOps.invert(img.resize((28, 28)))
            
            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0
            
            # Reshape for CNN input: (1, 28, 28, 1)
            cnn_input = img_array.reshape(1, 28, 28, 1)
            st.image(img, caption="Processed Drawing", width=150)
            try:
                prediction = model.predict(cnn_input)
                predicted_class = classes[np.argmax(prediction)]
                st.markdown(f'<div class="prediction"> Predicted: <strong>{predicted_class}</strong></div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")
    else: st.warning("Please add a letter to the canvas!")