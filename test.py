from tensorflow.keras.models import load_model

model_path = "plant_disease_model.h5"
print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully!")
