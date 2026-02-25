import os
import io
import base64
import threading
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
from tensorflow.keras.models import load_model

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- Config ---
MODEL_PATH = "plant_disease_model.h5"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

REMEDY_SUGGESTIONS = {
    'Tomato___Late_blight': {
        'organic': "Apply copper-based fungicides and remove infected plants.",
        'chemical': "Use chlorothalonil or mancozeb fungicides."
    },
    'Potato___Early_blight': {
        'organic': "Use neem oil and practice crop rotation.",
        'chemical': "Use fungicides containing mancozeb or azoxystrobin."
    },
    'default': {
        'organic': "Maintain healthy soil and remove diseased leaves.",
        'chemical': "Consult local agricultural experts for targeted treatment."
    }
}

model = None

# --- Helper Functions ---
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def format_class_name(name):
    formatted = name.replace("___", ": ").replace("_", " ")
    return formatted.title()

# --- Tkinter UI Setup ---
root = tk.Tk()
root.title("🌿 Plant Disease Detector")
root.geometry("650x700")
root.configure(bg="#f0f8f5")

title_label = tk.Label(root, text="🌿 Plant Disease Detector", font=("Helvetica", 20, "bold"), bg="#f0f8f5", fg="#2e7d32")
title_label.pack(pady=20)

status_label = tk.Label(root, text="Loading model... please wait ⏳", font=("Helvetica", 12), bg="#f0f8f5", fg="#666")
status_label.pack(pady=5)

canvas = tk.Canvas(root, width=400, height=300, bg="white", highlightthickness=2, highlightbackground="#4caf50")
canvas.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 13), bg="#f0f8f5", justify="left", wraplength=550)
result_label.pack(pady=20)

def load_model_thread():
    global model
    try:
        model = load_model(MODEL_PATH)
        status_label.config(text="✅ Model loaded successfully!", fg="#2e7d32")
        upload_button.config(state="normal")
    except Exception as e:
        status_label.config(text=f"❌ Failed to load model: {e}", fg="red")
        upload_button.config(state="disabled")

def select_image():
    if model is None:
        messagebox.showwarning("Model Not Ready", "Please wait until the model finishes loading.")
        return

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
    if not file_path:
        return

    img = Image.open(file_path)
    img.thumbnail((400, 300))
    tk_img = ImageTk.PhotoImage(img)
    canvas.image = tk_img
    canvas.create_image(200, 150, image=tk_img)

    result_label.config(text="Analyzing image... please wait ⏳", fg="#555")
    root.update()
    predict_image(file_path)

def predict_image(path):
    try:
        img_array = preprocess_image(path)
        preds = model.predict(img_array, verbose=0)
        class_index = np.argmax(preds)
        class_name = CLASS_NAMES[class_index]
        confidence = round(float(np.max(preds) * 100), 2)
        is_healthy = "healthy" in class_name.lower()

        result_text = f"🩺 Prediction: {format_class_name(class_name)}\n"
        result_text += f"📊 Confidence: {confidence}%\n"

        if is_healthy:
            result_text += "\n✅ This plant appears healthy! Keep good care practices."
        else:
            remedy = REMEDY_SUGGESTIONS.get(class_name, REMEDY_SUGGESTIONS['default'])
            result_text += f"\n🌱 Organic Treatment:\n{remedy['organic']}\n\n⚗️ Chemical Treatment:\n{remedy['chemical']}"

        result_label.config(text=result_text, fg="black")

    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error during prediction: {e}")

upload_button = tk.Button(root, text="📁 Select Plant Image", command=select_image,
                          bg="#4caf50", fg="white", font=("Helvetica", 12, "bold"),
                          relief="raised", padx=20, pady=10, borderwidth=3, state="disabled")
upload_button.pack(pady=15)

exit_button = tk.Button(root, text="❌ Exit", command=root.quit,
                        bg="#c62828", fg="white", font=("Helvetica", 12, "bold"),
                        relief="raised", padx=20, pady=10, borderwidth=3)
exit_button.pack(pady=10)

# Load model in background
threading.Thread(target=load_model_thread, daemon=True).start()

root.mainloop()
