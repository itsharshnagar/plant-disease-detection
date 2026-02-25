import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load your trained model
MODEL_PATH = "plant_disease_model.h5"  # Change path if needed
model = tf.keras.models.load_model(r"C:\Users\user\Desktop\Final Plant Detection\plant_disease_model.h5")

# Define image size (same as used during training)
IMG_SIZE = (224, 224)

# Create main window
root = tk.Tk()
root.title("Plant Disease Detector")
root.geometry("600x700")
root.configure(bg="#f2f2f2")

# Label to show selected image
img_label = Label(root, bg="#f2f2f2")
img_label.pack(pady=20)

# Label to show prediction
result_label = Label(root, text="Upload an image to predict", font=("Arial", 16), bg="#f2f2f2")
result_label.pack(pady=20)

# Function to preprocess image and predict
def predict_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    
    class_names = ['glioma', 'no_tumor', 'meningioma', 'pituitary']  # Change for your dataset
    predicted_class = class_names[predicted_class_index]
    confidence = np.max(prediction) * 100

    result_label.config(text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")

# Function to upload and show image
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", ".jpg;.jpeg;*.png")]
    )
    if not file_path:
        return

    img = Image.open(file_path)
    img = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    
    img_label.config(image=img_tk)
    img_label.image = img_tk

    predict_image(file_path)

# Upload button
upload_btn = Button(root, text="Upload Image", command=upload_image,
                    bg="#4CAF50", fg="white", font=("Arial", 14), padx=20, pady=10)
upload_btn.pack(pady=30)

# Run the Tkinter event loop
root.mainloop()