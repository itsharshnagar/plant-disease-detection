import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tkinter import filedialog, Tk

# --- Settings ---
MODEL_PATH = r"C:/Users/user\Desktop/Final Plant Detection/plant_disease_model.h5"  # your model file name
IMG_SIZE = (224, 224)  # change if your model expects another input size

# --- Load Model ---
print("🔹 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# --- Select Image ---
root = Tk()
root.withdraw()  # Hide Tkinter window
file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")]
)
if not file_path:
    print("❌ No image selected.")
    exit()

# --- Preprocess Image ---
img = Image.open(file_path).convert("RGB")
img = img.resize(IMG_SIZE)
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# --- Predict ---
print("🔹 Running prediction...")
preds = model.predict(img_array)
class_index = np.argmax(preds)
confidence = round(float(np.max(preds) * 100), 2)

print("\n✅ Prediction complete!")
print("Class Index:", class_index)
print("Confidence:", confidence, "%")
