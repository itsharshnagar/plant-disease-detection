import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="auto",
)

MODEL_PATH = 'plant_disease_model.h5'

CLASS_NAMES = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
    'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust',
    'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot',
    'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
    'Pepper,bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato__Early_blight',
    'Potato__Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean__healthy',
    'Squash__Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry__healthy',
    'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato__Late_blight',
    'Tomato__Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato___healthy'
]

REMEDY_SUGGESTIONS = {
    'Apple___Apple_scab': {
        'organic': "Apply sulfur-based fungicides or neem oil. Ensure good air circulation by pruning trees.",
        'chemical': "Use fungicides containing captan, myclobutanil, or mancozeb."
    },
    'Tomato___Late_blight': {
        'organic': "Apply copper-based fungicides. Ensure good air circulation and avoid overhead watering.",
        'chemical': "Use fungicides containing chlorothalonil, mancozeb, or metalaxyl."
    },
    'Potato___Late_blight': {
        'organic': "Apply copper-based fungicides. Remove and destroy infected plants.",
        'chemical': "Use fungicides like chlorothalonil or mancozeb."
    },
    'default': {
        'organic': "Maintain good garden hygiene, ensure proper watering and soil nutrition.",
        'chemical': "Consult a local agricultural extension for specific chemical recommendations."
    }
}

@st.cache_resource
def load_plant_model():
    """Load the pre-trained Keras model from the specified path."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: The model file '{MODEL_PATH}' was not found.")
        st.error("Please make sure the model file is in the same directory as this app.")
        return None
    try:
        # This print statement will show in your terminal, confirming the load has started
        print("--- Loading model from disk ---")
        model = load_model(MODEL_PATH)
        print("--- Model loaded successfully ---")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocesses the uploaded image to fit the model's input requirements."""
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# --- App Layout ---
st.title("🌿 AI-Powered Plant Disease Detection")
st.markdown("Upload an image of a plant leaf to identify potential diseases and receive remedy suggestions.")

# Load the model with a user-friendly spinner
with st.spinner("🚀 Loading the AI model... This may take a moment on first startup."):
    model = load_plant_model()

# Only show the uploader and the rest of the app if the model has loaded successfully
if model is not None:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")

        with st.spinner("Classifying..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = np.max(prediction) * 100

            st.write("---")
            st.header("🔬 Analysis Results")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Top Prediction", predicted_class_name.replace('', ' - ').replace('', ' '))
            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")

            if 'healthy' in predicted_class_name:
                st.balloons()
                st.success("Great news! The plant appears to be healthy.")
            else:
                with st.expander("💡 View Remedy Suggestions"):
                    remedy = REMEDY_SUGGESTIONS.get(predicted_class_name, REMEDY_SUGGESTIONS['default'])
                    st.subheader("Organic Approach")
                    st.markdown(f"<div style='background-color:#e8f5e9; padding:10px; border-radius:5px;'>{remedy['organic']}</div>", unsafe_allow_html=True)
                    st.subheader("Chemical Approach")
                    st.markdown(f"<div style='background-color:#ffebee; padding:10px; border-radius:5px;'>{remedy['chemical']}</div>", unsafe_allow_html=True)
else:
    st.stop() # Stop the app if the model could not be loaded

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "This application uses a deep learning model to identify plant diseases from leaf images. "
    "It's designed to help farmers and gardeners quickly diagnose and treat common ailments."
)
st.sidebar.warning(
    "This tool provides suggestions and is not a substitute for professional agricultural advice."
)