# import os
# # This line hides the informational oneDNN message from TensorFlow
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import io
# import base64
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from flask import Flask, request, render_template_string

# # --- Initialize the Flask App ---
# app = Flask(__name__)

# # --- Configure Model Path and Class Names ---
# MODEL_PATH = 'plant_disease_model.h5'
# CLASS_NAMES = [
#     'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
#     'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
#     'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust',
#     'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot',
#     'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#     'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
#     'Pepper,bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato__Early_blight',
#     'Potato__Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean__healthy',
#     'Squash__Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry__healthy',
#     'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato__Late_blight',
#     'Tomato__Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
#     'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
#     'Tomato___healthy'
# ]

# # --- Load the AI Model ---
# model = None
# def load_plant_model():
#     """Loads the trained model into memory."""
#     global model
#     try:
#         model = load_model(r"C:\Users\user\Desktop\Final Plant Detection\plant_disease_model.h5")
#         print("✅ Model loaded successfully!")
#     except Exception as e:
#         print(f"❌ Error loading model: {e}")

# # --- Image Preprocessing Function ---
# def preprocess_image(image_bytes):
#     """Takes image bytes and prepares them for the model."""
#     img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     img = img.resize((224, 224))
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0) # Create a batch
#     img_array = img_array / 255.0 # Normalize
#     return img_array

# # --- HTML & CSS Template ---
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Plant Disease Detector</title>
#     <style>
#         body { font-family: 'Segoe UI', sans-serif; background-color: #f4f7f6; margin: 0; padding: 2rem; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
#         .container { background: white; padding: 2.5rem; border-radius: 15px; box-shadow: 0 8px 30px rgba(0,0,0,0.12); width: 100%; max-width: 600px; text-align: center; }
#         h1 { color: #2a623d; }
#         p { color: #555; }
#         .upload-box { border: 2px dashed #ccc; padding: 2rem; border-radius: 10px; cursor: pointer; transition: background-color 0.2s; }
#         .upload-box:hover { background-color: #f9f9f9; }
#         input[type="file"] { display: none; }
#         label { font-weight: bold; color: #333; }
#         #preview { max-width: 50%; border-radius: 10px; margin-top: 1rem; }
#         .btn { background-color: #4CAF50; color: white; padding: 0.8rem 1.5rem; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; margin-top: 1rem; }
#         .btn:hover { background-color: #45a049; }
#         .results { margin-top: 2rem; text-align: left; }
#         .results h2 { text-align:center; color: #2a623d; }
#         .results p { background: #e8f5e9; padding: 1rem; border-radius: 8px; }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>🌿 Plant Disease Detector</h1>
#         <p>Upload a leaf image to get a diagnosis.</p>
#         <form method="post" enctype="multipart/form-data">
#             <div class="upload-box" onclick="document.getElementById('file-input').click();">
#                 <input type="file" name="file" id="file-input" accept="image/*" onchange="previewImage(event)">
#                 <label for="file-input">Click here to select an image</label>
#             </div>
#             <img id="preview" src="" alt="Image Preview" style="display:none;"/>
#             <button type="submit" class="btn">Diagnose</button>
#         </form>
#         {% if result %}
#         <div class="results">
#             <h2>Analysis Results</h2>
#             <p><strong>Prediction:</strong> {{ result.prediction }}</p>
#             <p><strong>Confidence:</strong> {{ result.confidence }}%</p>
#         </div>
#         {% endif %}
#     </div>
#     <script>
#         function previewImage(event) {
#             const reader = new FileReader();
#             reader.onload = function(){
#                 const preview = document.getElementById('preview');
#                 preview.src = reader.result;
#                 preview.style.display = 'block';
#             }
#             reader.readAsDataURL(event.target.files[0]);
#         }
#     </script>
# </body>
# </html>
# """

# # --- Flask Routes ---
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     result = None
#     if request.method == 'POST':
#         if 'file' in request.files:
#             file = request.files['file']
#             if file.filename != '' and model is not None:
#                 image_bytes = file.read()
#                 processed_image = preprocess_image(image_bytes)
#                 prediction = model.predict(processed_image)
                
#                 class_index = np.argmax(prediction)
#                 class_name = CLASS_NAMES[class_index]
#                 confidence = round(np.max(prediction) * 100, 2)
                
#                 result = {
#                     "prediction": class_name.replace('', ' - ').replace('', ' '),
#                     "confidence": confidence
#                 }
#     return render_template_string(HTML_TEMPLATE, result=result)

# # --- Main Execution ---
# if __name__ == '_main_':
#     # First, check if the model file exists before trying to load it
#     if not os.path.exists(MODEL_PATH):
#         print(f"❌ CRITICAL ERROR: The model file '{MODEL_PATH}' was not found.")
#         print("Please make sure the model is in the same directory as this script.")
#     else:
#         print("--- 🚀 Attempting to load AI model. This may take over a minute... ---")
#         load_plant_model()
        
#         # Only start the server if the model was loaded successfully
#         if model is not None:
#             print("--- Starting Flask server... ---")
#             app.run(port=5000)

# import os
# # This line hides the informational oneDNN message from TensorFlow
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import io
# import base64
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from flask import Flask, request, render_template_string

# # --- Initialize the Flask App ---
# app = Flask(__name__)

# # --- Configure Model Path, Class Names, and Remedies ---
# MODEL_PATH = 'plant_disease_model.h5'
# CLASS_NAMES = [
#     'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
#     'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
#     'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust',
#     'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot',
#     'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#     'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
#     'Pepper,bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato__Early_blight',
#     'Potato__Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean__healthy',
#     'Squash__Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry__healthy',
#     'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato__Late_blight',
#     'Tomato__Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
#     'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
#     'Tomato___healthy'
# ]
# REMEDY_SUGGESTIONS = {
#     'Apple___Apple_scab': {
#         'organic': "Apply sulfur-based fungicides or neem oil. Ensure good air circulation by pruning trees.",
#         'chemical': "Use fungicides containing captan, myclobutanil, or mancozeb."
#     },
#     'Tomato___Late_blight': {
#         'organic': "Apply copper-based fungicides. Ensure good air circulation and avoid overhead watering.",
#         'chemical': "Use fungicides containing chlorothalonil, mancozeb, or metalaxyl."
#     },
#     'Potato___Late_blight': {
#         'organic': "Apply copper-based fungicides. Remove and destroy infected plants.",
#         'chemical': "Use fungicides like chlorothalonil or mancozeb."
#     },
#     'default': {
#         'organic': "Maintain good garden hygiene, ensure proper watering and soil nutrition.",
#         'chemical': "Consult a local agricultural extension for specific chemical recommendations."
#     }
# }


# # --- Load the AI Model ---
# model = None
# def load_plant_model():
#     """Loads the trained model into memory."""
#     global model
#     try:
#         model = load_model(MODEL_PATH)
#         print("✅ Model loaded successfully!")
#     except Exception as e:
#         print(f"❌ Error loading model: {e}")

# # --- Image Preprocessing Function ---
# def preprocess_image(image_bytes):
#     """Takes image bytes and prepares them for the model."""
#     img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     img = img.resize((224, 224))
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0) # Create a batch
#     img_array = img_array / 255.0 # Normalize
#     return img_array

# # --- HTML & CSS Template ---
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Plant Disease Detector</title>
#     <style>
#         body { font-family: 'Segoe UI', sans-serif; background-color: #f4f7f6; margin: 0; padding: 2rem; display: flex; justify-content: center; align-items: center; min-height: 100vh; flex-direction: column; }
#         .container { background: white; padding: 2.5rem; border-radius: 15px; box-shadow: 0 8px
#  30px rgba(0,0,0,0.12); width: 100%; max-width: 600px; text-align: center; margin-bottom: 2rem; }
#         h1 { color: #2a623d; }
#         p { color: #555; }
#         .upload-box { border: 2px dashed #ccc; padding: 2rem; border-radius: 10px; cursor: pointer; transition: background-color 0.2s; }
#         .upload-box:hover { background-color: #f9f9f9; }
#         input[type="file"] { display: none; }
#         label { font-weight: bold; color: #333; cursor: pointer; }
#         #preview { max-width: 50%; border-radius: 10px; margin-top: 1rem; }
#         .btn { background-color: #4CAF50; color: white; padding: 0.8rem 1.5rem; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; margin-top: 1rem; }
#         .btn:hover { background-color: #45a049; }
#         .loader { display: none; width: 40px; height: 40px; border: 4px solid #f3f3f3; border-top: 4px solid #4CAF50; border-radius: 50%; animation: spin 1s linear infinite; margin: 1rem auto 0 auto; }
#         @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
#         .results-card { text-align: left; background-color: #e8f5e9; padding: 1.5rem; border-left: 5px solid #4CAF50; border-radius: 8px; }
#         .results-card h2 { margin-top: 0; color: #2a623d; }
#         .results-card p { margin: 0.5rem 0; }
#         .remedy { margin-top: 1rem; }
#         .remedy h3 { font-size: 1rem; color: #333; border-top: 1px solid #ccc; padding-top: 1rem; margin-top: 1rem; }
#         .remedy p { background: #fff; padding: 0.8rem; border-radius: 6px; }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>🌿 Plant Disease Detector</h1>
#         <p>Upload a leaf image to get a diagnosis and remedy suggestions.</p>
#         <form id="upload-form" method="post" enctype="multipart/form-data">
#             <div class="upload-box" onclick="document.getElementById('file-input').click();">
#                 <input type="file" name="file" id="file-input" accept="image/*" onchange="previewImage(event)">
#                 <label for="file-input">Click here to select an image</label>
#             </div>
#             <img id="preview" src="" alt="Image Preview" style="display:none;"/>
#             <button type="submit" class="btn">Diagnose</button>
#         </form>
#         <div class="loader" id="loader"></div>
#     </div>

#     {% if result %}
#     <div class="container results-card">
#         <h2>🔬 Analysis Results</h2>
#         <p><strong>Prediction:</strong> {{ result.prediction }}</p>
#         <p><strong>Confidence:</strong> {{ result.confidence }}%</p>
#         <img src="data:image/jpeg;base64,{{ result.image }}" style="max-width:100%; margin-top:1rem; border-radius:10px;" alt="Uploaded Image">
        
#         {% if result.remedy %}
#         <div class="remedy">
#             <h3>Organic Remedy</h3>
#             <p>{{ result.remedy.organic }}</p>
#             <h3>Chemical Remedy</h3>
#             <p>{{ result.remedy.chemical }}</p>
#         </div>
#         {% else %}
#         <div class="remedy">
#             <h3>✅ Healthy Plant</h3>
#             <p>This plant appears to be healthy. Continue with good care practices.</p>
#         </div>
#         {% endif %}
#     </div>
#     {% endif %}

#     <script>
#         const uploadForm = document.getElementById('upload-form');
#         const loader = document.getElementById('loader');
        
#         function previewImage(event) {
#             const reader = new FileReader();
#             reader.onload = function(){
#                 const preview = document.getElementById('preview');
#                 preview.src = reader.result;
#                 preview.style.display = 'block';
#             }
#             reader.readAsDataURL(event.target.files[0]);
#         }
        
#         uploadForm.addEventListener('submit', function() {
#             loader.style.display = 'block';
#         });
#     </script>
# </body>
# </html>
# """

# # --- Flask Routes ---
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     result = None
#     if request.method == 'POST':
#         if 'file' in request.files:
#             file = request.files['file']
#             if file.filename != '' and model is not None:
#                 image_bytes = file.read()
#                 processed_image = preprocess_image(image_bytes)
#                 prediction = model.predict(processed_image)
                
#                 class_index = np.argmax(prediction)
#                 class_name = CLASS_NAMES[class_index]
#                 confidence = round(np.max(prediction) * 100, 2)
                
#                 # Check if the plant is healthy to decide whether to show a remedy
#                 is_healthy = 'healthy' in class_name
#                 remedy = REMEDY_SUGGESTIONS.get(class_name, REMEDY_SUGGESTIONS['default']) if not is_healthy else None

#                 result = {
#                     "prediction": class_name.replace('', ' - ').replace('', ' '),
#                     "confidence": confidence,
#                     "remedy": remedy,
#                     "image": base64.b64encode(image_bytes).decode('utf-8')
#                 }
#     return render_template_string(HTML_TEMPLATE, result=result)

# # --- Main Execution ---
# if __name__ == '_main_':
#     if not os.path.exists(MODEL_PATH):
#         print(f"❌ CRITICAL ERROR: The model file '{MODEL_PATH}' was not found.")
#     else:
#         print("--- 🚀 Attempting to load AI model. This may take over a minute... ---")
#         load_plant_model()
#         if model is not None:
#             print("--- Starting Flask server... ---")
#             app.run(port=5000)

# import os
# # This line hides the informational oneDNN message from TensorFlow
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import io
# import base64
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from flask import Flask, request, render_template_string

# # --- Initialize the Flask App ---
# app = Flask(__name__)

# # --- Configure Model Path, Class Names, and Remedies ---
# MODEL_PATH = 'plant_disease_model.h5'
# CLASS_NAMES = [
#     'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
#     'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
#     'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust',
#     'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot',
#     'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#     'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
#     'Pepper,bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato__Early_blight',
#     'Potato__Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean__healthy',
#     'Squash__Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry__healthy',
#     'Tomato__Bacterial_spot', 'Tomato_Early_blight', 'Tomato__Late_blight',
#     'Tomato__Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
#     'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
#     'Tomato___healthy'
# ]
# REMEDY_SUGGESTIONS = {
#     'Apple___Apple_scab': {
#         'organic': "Apply sulfur-based fungicides or neem oil. Ensure good air circulation by pruning trees.",
#         'chemical': "Use fungicides containing captan, myclobutanil, or mancozeb."
#     },
#     'Tomato___Late_blight': {
#         'organic': "Apply copper-based fungicides. Ensure good air circulation and avoid overhead watering.",
#         'chemical': "Use fungicides containing chlorothalonil, mancozeb, or metalaxyl."
#     },
#     'Potato___Late_blight': {
#         'organic': "Apply copper-based fungicides. Remove and destroy infected plants.",
#         'chemical': "Use fungicides like chlorothalonil or mancozeb."
#     },
#     'default': {
#         'organic': "Maintain good garden hygiene, ensure proper watering and soil nutrition.",
#         'chemical': "Consult a local agricultural extension for specific chemical recommendations."
#     }
# }


# # --- Load the AI Model ---
# model = None
# def load_plant_model():
#     """Loads the trained model into memory."""
#     global model
#     try:
#         model = load_model(MODEL_PATH)
#         print("✅ Model loaded successfully!")
#     except Exception as e:
#         print(f"❌ Error loading model: {e}")

# # --- Image Preprocessing Function ---
# def preprocess_image(image_bytes):
#     """Takes image bytes and prepares them for the model."""
#     img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#     img = img.resize((224, 224))
#     img_array = np.array(img)
#     img_array = np.expand_dims(img_array, axis=0) # Create a batch
#     img_array = img_array / 255.0 # Normalize
#     return img_array

# # --- HTML & CSS Template ---
# HTML_TEMPLATE = """
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Plant Disease Detector</title>
#     <style>
#         body { font-family: 'Segoe UI', sans-serif; background-color: #f4f7f6; margin: 0; padding: 2rem; display: flex; justify-content: center; align-items: center; min-height: 100vh; flex-direction: column; }
#         .container { background: white; padding: 2.5rem; border-radius: 15px; box-shadow: 0 8px
#  30px rgba(0,0,0,0.12); width: 100%; max-width: 600px; text-align: center; margin-bottom: 2rem; }
#         h1 { color: #2a623d; }
#         p { color: #555; }
#         .upload-box { border: 2px dashed #ccc; padding: 2rem; border-radius: 10px; cursor: pointer; transition: background-color 0.2s; }
#         .upload-box:hover { background-color: #f9f9f9; }
#         input[type="file"] { display: none; }
#         label { font-weight: bold; color: #333; cursor: pointer; }
#         #preview { max-width: 50%; border-radius: 10px; margin-top: 1rem; }
#         .btn { background-color: #4CAF50; color: white; padding: 0.8rem 1.5rem; border: none; border-radius: 8px; cursor: pointer; font-size: 1rem; margin-top: 1rem; }
#         .btn:hover { background-color: #45a049; }
#         .loader { display: none; width: 40px; height: 40px; border: 4px solid #f3f3f3; border-top: 4px solid #4CAF50; border-radius: 50%; animation: spin 1s linear infinite; margin: 1rem auto 0 auto; }
#         @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
#         .results-card { text-align: left; background-color: #e8f5e9; padding: 1.5rem; border-left: 5px solid #4CAF50; border-radius: 8px; }
#         .results-card h2 { margin-top: 0; color: #2a623d; }
#         .results-card p { margin: 0.5rem 0; }
#         .remedy { margin-top: 1rem; }
#         .remedy h3 { font-size: 1rem; color: #333; border-top: 1px solid #ccc; padding-top: 1rem; margin-top: 1rem; }
#         .remedy p { background: #fff; padding: 0.8rem; border-radius: 6px; }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>🌿 Plant Disease Detector</h1>
#         <p>Upload a leaf image to get a diagnosis and remedy suggestions.</p>
#         <form id="upload-form" method="post" enctype="multipart/form-data">
#             <div class="upload-box" onclick="document.getElementById('file-input').click();">
#                 <input type="file" name="file" id="file-input" accept="image/*" onchange="previewImage(event)">
#                 <label for="file-input">Click here to select an image</label>
#             </div>
#             <img id="preview" src="" alt="Image Preview" style="display:none;"/>
#             <button type="submit" class="btn">Diagnose</button>
#         </form>
#         <div class="loader" id="loader"></div>
#     </div>

#     {% if result %}
#     <div class="container results-card">
#         <h2>🔬 Analysis Results</h2>
#         <p><strong>Prediction:</strong> {{ result.prediction }}</p>
#         <p><strong>Confidence:</strong> {{ result.confidence }}%</p>
#         <img src="data:image/jpeg;base64,{{ result.image }}" style="max-width:100%; margin-top:1rem; border-radius:10px;" alt="Uploaded Image">
        
#         {% if result.remedy %}
#         <div class="remedy">
#             <h3>Organic Remedy</h3>
#             <p>{{ result.remedy.organic }}</p>
#             <h3>Chemical Remedy</h3>
#             <p>{{ result.remedy.chemical }}</p>
#         </div>
#         {% else %}
#         <div class="remedy">
#             <h3>✅ Healthy Plant</h3>
#             <p>This plant appears to be healthy. Continue with good care practices.</p>
#         </div>
#         {% endif %}
#     </div>
#     {% endif %}

#     <script>
#         const uploadForm = document.getElementById('upload-form');
#         const loader = document.getElementById('loader');
        
#         function previewImage(event) {
#             const reader = new FileReader();
#             reader.onload = function(){
#                 const preview = document.getElementById('preview');
#                 preview.src = reader.result;
#                 preview.style.display = 'block';
#             }
#             reader.readAsDataURL(event.target.files[0]);
#         }
        
#         uploadForm.addEventListener('submit', function() {
#             loader.style.display = 'block';
#         });
#     </script>
# </body>
# </html>
# """

# # --- Flask Routes ---
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     result = None
#     if request.method == 'POST':
#         # Check if a file was uploaded
#         if 'file' not in request.files or request.files['file'].filename == '':
#             return render_template_string(HTML_TEMPLATE)

#         file = request.files['file']
        
#         # Only proceed if the model is loaded
#         if model is not None:
#             image_bytes = file.read()
#             processed_image = preprocess_image(image_bytes)
#             prediction = model.predict(processed_image)
            
#             class_index = np.argmax(prediction)
#             class_name = CLASS_NAMES[class_index]
#             confidence = round(np.max(prediction) * 100, 2)
            
#             is_healthy = 'healthy' in class_name
#             remedy = REMEDY_SUGGESTIONS.get(class_name, REMEDY_SUGGESTIONS['default']) if not is_healthy else None

#             result = {
#                 "prediction": class_name.replace('', ' - ').replace('', ' '),
#                 "confidence": confidence,
#                 "remedy": remedy,
#                 "image": base64.b64encode(image_bytes).decode('utf-8')
#             }
#     return render_template_string(HTML_TEMPLATE, result=result)

# # --- Main Execution ---
# if __name__ == '_main_':
#     # First, check if the model file exists before trying to load it
#     if not os.path.exists(MODEL_PATH):
#         print(f"❌ CRITICAL ERROR: The model file '{MODEL_PATH}' was not found.")
#         print("Please make sure the model is in the same directory as this script.")
#     else:
#         print("--- 🚀 Attempting to load AI model. This may take over a minute... ---")
#         load_plant_model()
        
#         # Only start the server if the model was loaded successfully
#         if model is not None:
#             print("--- Starting Flask server... ---")
#             app.run(port=5000) # Running on port 5000 as shown in your screenshot
#         else:
#              print("--- Flask server did not start because the model failed to load. ---")
import os
# This line hides the informational oneDNN message from TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template_string, flash, session, redirect, url_for
from werkzeug.utils import secure_filename

# --- Initialize the Flask App ---
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # For flash messages
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# --- Configure Model Path, Class Names, and Remedies ---
MODEL_PATH = 'plant_disease_model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

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
    'Apple___Apple_scab': {
        'organic': "Apply sulfur-based fungicides or neem oil. Ensure good air circulation by pruning trees. Remove fallen leaves to reduce spore sources.",
        'chemical': "Use fungicides containing captan, myclobutanil, or mancozeb. Apply preventatively during wet spring weather."
    },
    'Apple___Black_rot': {
        'organic': "Remove and destroy infected fruit and branches. Apply copper-based fungicides during dormant season.",
        'chemical': "Use fungicides containing captan or thiophanate-methyl."
    },
    'Tomato___Late_blight': {
        'organic': "Apply copper-based fungicides. Ensure good air circulation and avoid overhead watering. Remove infected plants immediately.",
        'chemical': "Use fungicides containing chlorothalonil, mancozeb, or metalaxyl. Apply at first sign of disease."
    },
    'Tomato___Early_blight': {
        'organic': "Remove affected leaves, improve air circulation, and apply copper fungicides or neem oil.",
        'chemical': "Use fungicides containing chlorothalonil or mancozeb."
    },
    'Potato___Late_blight': {
        'organic': "Apply copper-based fungicides. Remove and destroy infected plants. Avoid overhead watering.",
        'chemical': "Use fungicides like chlorothalonil or mancozeb. Monitor weather for blight-favorable conditions."
    },
    'Potato___Early_blight': {
        'organic': "Practice crop rotation, remove infected foliage, and apply copper-based sprays.",
        'chemical': "Use fungicides containing chlorothalonil, mancozeb, or azoxystrobin."
    },
    'default': {
        'organic': "Maintain good garden hygiene, ensure proper watering and soil nutrition. Remove diseased plant material promptly.",
        'chemical': "Consult a local agricultural extension office for specific chemical recommendations based on your region."
    }
}

# --- Load the AI Model ---
model = None
model_load_error = None

def load_plant_model():
    """Loads the trained model into memory."""
    global model, model_load_error
    try:
        print("Attempting to load model...")
        print(f"Model path: {os.path.abspath(MODEL_PATH)}")
        
        if not os.path.exists(MODEL_PATH):
            model_load_error = f"Model file '{MODEL_PATH}' not found in directory: {os.getcwd()}"
            print(f"❌ {model_load_error}")
            return False
        
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        model_load_error = None
        return True
    except FileNotFoundError:
        model_load_error = f"Model file not found at '{MODEL_PATH}'"
        print(f"❌ Error: {model_load_error}")
        print(f"Current working directory: {os.getcwd()}")
        return False
    except Exception as e:
        model_load_error = f"{type(e).__name__}: {str(e)}"
        print(f"❌ Error loading model: {model_load_error}")
        import traceback
        traceback.print_exc()
        return False

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    """Takes image bytes and prepares them for the model."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def format_class_name(class_name):
    """Format the class name for better readability."""
    # Replace underscores with spaces and clean up formatting
    formatted = class_name.replace('___', ': ').replace('_', ' ')
    # Capitalize appropriately
    parts = formatted.split(': ')
    if len(parts) == 2:
        return f"{parts[0]}: {parts[1].title()}"
    return formatted.title()

# --- Login HTML Template ---
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Plant Disease Detector</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            color: #333;
        }
        .login-container {
            background: white;
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 400px;
            text-align: center;
        }
        .login-container h1 {
            color: #2a623d;
            margin-bottom: 1.5rem;
            font-size: 2rem;
        }
        .login-container p {
            color: #666;
            margin-bottom: 2rem;
        }
        .form-group {
            margin-bottom: 1.5rem;
            text-align: left;
        }
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: bold;
            color: #333;
        }
        .form-group input {
            width: 100%;
            padding: 0.8rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        .form-group input:focus {
            outline: none;
            border-color: #4CAF50;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 0.8rem 2rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            width: 100%;
            transition: background-color 0.3s ease;
            font-weight: 600;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .btn:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .forgot-password {
            margin-top: 1rem;
            font-size: 0.9rem;
        }
        .forgot-password a {
            color: #4CAF50;
            text-decoration: none;
        }
        .forgot-password a:hover {
            text-decoration: underline;
        }
        .register-link {
            margin-top: 1.5rem;
            font-size: 0.9rem;
        }
        .register-link a {
            color: #667eea;
            text-decoration: none;
        }
        .register-link a:hover {
            text-decoration: underline;
        }
        .error-message {
            color: #d32f2f;
            background-color: #ffebee;
            padding: 0.5rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>🌿 Login</h1>
        <p>Welcome back! Please sign in to access the Plant Disease Detector.</p>

        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                <div class="error-message">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <form id="login-form" action="/login" method="post">
            <div class="form-group">
                <label for="username">Username or Email</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit" class="btn">Sign In</button>
        </form>

        <div class="forgot-password">
            <a href="#">Forgot your password?</a>
        </div>

        <div class="register-link">
            Don't have an account? <a href="#">Register here</a>
        </div>
    </div>
</body>
</html>
"""

# --- HTML & CSS Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease Detector</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0; 
            padding: 2rem; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 100vh; 
            flex-direction: column; 
        }
        .container { 
            background: white; 
            padding: 2.5rem; 
            border-radius: 15px; 
            box-shadow: 0 10px 40px rgba(0,0,0,0.3); 
            width: 100%; 
            max-width: 600px; 
            text-align: center; 
            margin-bottom: 2rem; 
        }
        h1 { 
            color: #2a623d; 
            margin-top: 0;
            font-size: 2rem;
        }
        p { color: #555; }
        .upload-box { 
            border: 3px dashed #4CAF50; 
            padding: 2rem; 
            border-radius: 10px; 
            cursor: pointer; 
            transition: all 0.3s ease;
            background-color: #f9f9f9;
        }
        .upload-box:hover { 
            background-color: #e8f5e9;
            border-color: #45a049;
        }
        input[type="file"] { display: none; }
        label { 
            font-weight: bold; 
            color: #333; 
            cursor: pointer;
            display: block;
        }
        #preview { 
            max-width: 100%; 
            max-height: 300px;
            border-radius: 10px; 
            margin-top: 1rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .btn { 
            background-color: #4CAF50; 
            color: white; 
            padding: 0.8rem 2rem; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer; 
            font-size: 1rem; 
            margin-top: 1rem;
            transition: background-color 0.3s ease;
            font-weight: 600;
        }
        .btn:hover { background-color: #45a049; }
        .btn:disabled { 
            background-color: #ccc; 
            cursor: not-allowed; 
        }
        .loader { 
            display: none; 
            width: 40px; 
            height: 40px; 
            border: 4px solid #f3f3f3; 
            border-top: 4px solid #4CAF50; 
            border-radius: 50%; 
            animation: spin 1s linear infinite; 
            margin: 1rem auto 0 auto; 
        }
        @keyframes spin { 
            0% { transform: rotate(0deg); } 
            100% { transform: rotate(360deg); } 
        }
        .results-card { 
            text-align: left; 
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            padding: 2rem; 
            border-left: 5px solid #4CAF50; 
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .results-card h2 { 
            margin-top: 0; 
            color: #2a623d;
            font-size: 1.5rem;
        }
        .results-card p { 
            margin: 0.8rem 0; 
            font-size: 1.05rem;
        }
        .confidence-bar {
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.5s ease;
        }
        .remedy { 
            margin-top: 1.5rem; 
        }
        .remedy h3 { 
            font-size: 1.1rem; 
            color: #2a623d; 
            border-top: 2px solid #4CAF50; 
            padding-top: 1rem; 
            margin-top: 1rem; 
        }
        .remedy p { 
            background: #fff; 
            padding: 1rem; 
            border-radius: 8px;
            line-height: 1.6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .healthy-badge {
            display: inline-block;
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 1rem;
        }
        .disease-badge {
            display: inline-block;
            background-color: #ff9800;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 1rem;
        }
        .alert {
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 8px;
            background-color: #ffebee;
            color: #c62828;
            border-left: 4px solid #c62828;
        }
        .file-name {
            margin-top: 1rem;
            color: #666;
            font-size: 0.9rem;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌿 Plant Disease Detector</h1>
        <p>Upload a clear image of a plant leaf to get an AI-powered diagnosis and remedy suggestions.</p>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                <div class="alert">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <form id="upload-form" method="post" enctype="multipart/form-data">
            <div class="upload-box" onclick="document.getElementById('file-input').click();">
                <input type="file" name="file" id="file-input" accept="image/*" onchange="previewImage(event)" required>
                <label for="file-input">📁 Click here to select an image</label>
                <p style="font-size: 0.85rem; color: #888; margin-top: 0.5rem;">Supported formats: JPG, PNG, GIF, BMP, WebP</p>
            </div>
            <div class="file-name" id="file-name"></div>
            <img id="preview" src="" alt="Image Preview" style="display:none;"/>
            <button type="submit" class="btn" id="submit-btn">🔬 Diagnose Plant</button>
        </form>
        <div class="loader" id="loader"></div>
    </div>

    {% if result %}
    <div class="container results-card">
        <h2>🔬 Analysis Results</h2>
        
        {% if result.is_healthy %}
            <span class="healthy-badge">✅ Healthy Plant</span>
        {% else %}
            <span class="disease-badge">⚠️ Disease Detected</span>
        {% endif %}
        
        <p><strong>Diagnosis:</strong> {{ result.prediction }}</p>
        <p><strong>Confidence Level:</strong> {{ result.confidence }}%</p>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {{ result.confidence }}%;"></div>
        </div>
        
        <img src="data:image/jpeg;base64,{{ result.image }}" style="max-width:100%; margin-top:1.5rem; border-radius:10px;" alt="Analyzed Plant Image">
        
        {% if result.remedy %}
        <div class="remedy">
            <h3>🌱 Organic Treatment Options</h3>
            <p>{{ result.remedy.organic }}</p>
            <h3>⚗️ Chemical Treatment Options</h3>
            <p>{{ result.remedy.chemical }}</p>
        </div>
        {% else %}
        <div class="remedy">
            <h3>✅ Plant Care Recommendations</h3>
            <p>This plant appears to be healthy! Continue with good care practices including proper watering, adequate sunlight, and regular monitoring for any changes.</p>
        </div>
        {% endif %}
        
        <div style="margin-top: 2rem; text-align: center;">
            <a href="/" class="btn">🔄 Analyze Another Image</a>
        </div>
    </div>
    {% endif %}

    <script>
        const uploadForm = document.getElementById('upload-form');
        const loader = document.getElementById('loader');
        const submitBtn = document.getElementById('submit-btn');
        const fileNameDisplay = document.getElementById('file-name');
        
        function previewImage(event) {
            const file = event.target.files[0];
            if (file) {
                // Display file name
                fileNameDisplay.textContent = `Selected: ${file.name}`;
                
                // Preview image
                const reader = new FileReader();
                reader.onload = function(){
                    const preview = document.getElementById('preview');
                    preview.src = reader.result;
                    preview.style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        }
        
        uploadForm.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('file-input');
            if (!fileInput.files.length) {
                e.preventDefault();
                alert('Please select an image file first!');
                return;
            }
            
            loader.style.display = 'block';
            submitBtn.disabled = true;
            submitBtn.textContent = 'Analyzing...';
        });
    </script>
</body>
</html>
"""

# --- Flask Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Simple authentication (replace with your own logic)
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')

    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    result = None
    
    if request.method == 'POST':
        # Check if model is loaded
        if model is None:
            print("❌ POST request received but model is None")
            error_msg = '⚠️ AI Model is not loaded.'
            if model_load_error:
                error_msg += f' Error: {model_load_error}'
            else:
                error_msg += ' Please check the server console for errors and restart the application.'
            flash(error_msg)
            return render_template_string(HTML_TEMPLATE)
        
        print("✅ Model is loaded, processing request...")
        
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file uploaded. Please select an image.')
            return render_template_string(HTML_TEMPLATE)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected. Please choose an image.')
            return render_template_string(HTML_TEMPLATE)
        
        # Validate file extension
        if not allowed_file(file.filename):
            flash(f'Invalid file type. Please upload: {", ".join(ALLOWED_EXTENSIONS)}')
            return render_template_string(HTML_TEMPLATE)
        
        try:
            print(f"Processing file: {file.filename}")
            
            # Read and process image
            image_bytes = file.read()
            print(f"Image size: {len(image_bytes)} bytes")
            
            processed_image = preprocess_image(image_bytes)
            
            if processed_image is None:
                flash('Error processing image. Please try another image.')
                return render_template_string(HTML_TEMPLATE)
            
            print("Image preprocessed successfully, making prediction...")
            
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)
            
            class_index = np.argmax(prediction)
            class_name = CLASS_NAMES[class_index]
            confidence = round(float(np.max(prediction) * 100), 2)
            
            print(f"Prediction: {class_name} (confidence: {confidence}%)")
            
            # Check if plant is healthy
            is_healthy = 'healthy' in class_name.lower()
            
            # Get remedy suggestions
            remedy = None if is_healthy else REMEDY_SUGGESTIONS.get(
                class_name, 
                REMEDY_SUGGESTIONS['default']
            )
            
            result = {
                "prediction": format_class_name(class_name),
                "confidence": confidence,
                "remedy": remedy,
                "is_healthy": is_healthy,
                "image": base64.b64encode(image_bytes).decode('utf-8')
            }
            
            print("✅ Analysis complete!")
            
        except Exception as e:
            print(f"❌ Error during prediction: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            flash('An error occurred during analysis. Please try again.')
            return render_template_string(HTML_TEMPLATE)
    
    return render_template_string(HTML_TEMPLATE, result=result)

@app.route('/status')
def status():
    """Debug endpoint to check model status"""
    return {
        'model_loaded': model is not None,
        'model_load_error': model_load_error,
        'model_path': MODEL_PATH,
        'model_path_absolute': os.path.abspath(MODEL_PATH),
        'model_exists': os.path.exists(MODEL_PATH),
        'current_directory': os.getcwd(),
        'files_in_directory': os.listdir('.'),
        'h5_files': [f for f in os.listdir('.') if f.endswith('.h5')]
    }

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 16MB.')
    return render_template_string(HTML_TEMPLATE), 413

# --- Main Execution ---
if __name__ == '__main__':
    print("=" * 60)
    print("🌿 Plant Disease Detector - Starting Application")
    print("=" * 60)
    
    # Check if the model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ CRITICAL ERROR: Model file '{MODEL_PATH}' not found!")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📁 Files in current directory: {os.listdir('.')}")
        print("\nPlease ensure the model file is in the same directory as this script.")
        print("Looking for file named: plant_disease_model.h5")
        print("\n" + "=" * 60)
        
        # Still start the server but warn user
        print("\n⚠️  Starting server anyway for debugging...")
        print("You can try uploading the model file while the server is running.")
        print("=" * 60 + "\n")
    else:
        print(f"\n✅ Model file found: {MODEL_PATH}")
        print(f"📊 File size: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
        print("🔄 Loading AI model (this may take 1-2 minutes)...\n")
        
        if not load_plant_model():
            print("\n⚠️  Warning: Model failed to load, but starting server anyway...")
            print("Check the error messages above for details.\n")
    
    print("=" * 60)
    print("🚀 Starting Flask web server...")
    print("🌐 Access the application at: http://localhost:5000")
    print(f"🤖 Model status: {'✅ LOADED' if model is not None else '❌ NOT LOADED'}")
    print("📌 Press CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)