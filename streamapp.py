import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

def set_sea_background():
    st.markdown(
        """
        <style>
        /* Background image for the whole app */
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #ffffff;  /* Light text for better contrast */
        }

        /* Translucent background for text/content block */
        .block-container {
            background-color: rgba(0, 0, 0, 0.6);  /* semi-transparent dark */
            padding: 2rem;
            border-radius: 12px;
        }

        /* Center and style the title */
        .centered-title h1 {
            text-align: center;
            font-size: 3em;
            color: #ffffff;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_sea_background()

# Centered and styled title
st.markdown('<div class="centered-title"><h1>🐟 Fish Species Classifier (InceptionV3)</h1></div>', unsafe_allow_html=True)



from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("inception_model.h5")

# Load class labels
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

model = load_model()
img_size = (292, 292)

# upload prompt
st.markdown("Upload a fish image to identify its species using the deep learning model.")


# Custom label
st.markdown(
    "<p style='color: #222222; font-weight: bold; font-size: 18px;'>📤 Choose a fish🐟 image...</p>",
    unsafe_allow_html=True
)

# File uploader (no built-in label)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # Preprocess image
    image = image.resize(img_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Predict
    preds = model.predict(image_array)
    predicted_index = np.argmax(preds)
    predicted_class = idx_to_class[predicted_index]
    confidence = float(np.max(preds)) * 100

    # Display result
    st.markdown(
    f"""
    <div style='
        background-color: #fff3e0;  /* light orange */
        padding: 15px;
        border-left: 5px solid #4CAF50;  /* green accent */
        border-radius: 8px;
        color: #4CAF50;  /* green text */
        font-size: 18px;
        font-weight: bold;
    '>
    🎯 Predicted: {predicted_class} ({confidence:.2f}% confidence)
    </div>
    """,
    unsafe_allow_html=True
)


    # Show top-3 predictions (optional)
    top_3_indices = preds[0].argsort()[-3:][::-1]
    st.subheader("🔍 Top-3 Predictions:")
    for i in top_3_indices:
        st.write(f"- {idx_to_class[i]}: {preds[0][i]*100:.2f}%")

    st.markdown("---")
    st.markdown("**📌 Model: InceptionV3 | Trained on custom fish dataset | Built using Streamlit**")