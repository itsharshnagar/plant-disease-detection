import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Hides the oneDNN informational message
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from app4 import Flask, request, render_template_string

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit

# --- Model and App Configuration ---
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

# --- Load the Model ---
model = None
def load_plant_model():
    global model
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure 'plant_disease_model.h5' is in the same directory.")

# --- Helper Functions ---
def preprocess_image(image_bytes):
    """Preprocesses the image to fit the model's input requirements."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# --- HTML & CSS Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease Detection</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f0f4f8;
            color: #333;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            width: 100%;
            max-width: 700px;
            background-color: #ffffff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }
        h1 {
            color: #2c5282;
            margin-bottom: 10px;
        }
        p {
            color: #5a7b9d;
            margin-bottom: 30px;
        }
        .upload-area {
            border: 2px dashed #cbd5e0;
            border-radius: 8px;
            padding: 40px;
            cursor: pointer;
            transition: background-color 0.3s, border-color 0.3s;
        }
        .upload-area:hover {
            background-color: #e2e8f0;
            border-color: #4a5568;
        }
        .upload-area input[type="file"] {
            display: none;
        }
        .upload-area label {
            font-weight: 600;
            color: #4a5568;
            cursor: pointer;
        }
        #preview {
            margin-top: 20px;
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
            display: none;
        }
        .btn {
            background-color: #4299e1;
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
            transition: background-color 0.3s;
            display: none;
        }
        .btn:hover {
            background-color: #3182ce;
        }
        .loader {
            border: 4px solid #f3f3f3;
            border-radius: 50%;
            border-top: 4px solid #4299e1;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
            display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .results-card {
            margin-top: 30px;
            padding: 25px;
            background-color: #edf2f7;
            border-radius: 8px;
            text-align: left;
        }
        .results-card h2 {
            margin-top: 0;
            color: #2d3748;
            border-bottom: 2px solid #cbd5e0;
            padding-bottom: 10px;
        }
        .result-item {
            display: flex;
            justify-content: space-between;
            font-size: 1.1em;
            margin-bottom: 10px;
        }
        .result-item span:first-child {
            font-weight: 600;
        }
        .remedy {
            margin-top: 20px;
        }
        .remedy h3 {
            color: #4a5568;
        }
        .remedy p {
            background-color: #fff;
            padding: 15px;
            border-radius: 6px;
            color: #4a5568;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌿 Plant Disease Detector</h1>
        <p>Upload an image of a plant leaf to identify the disease and get remedy suggestions.</p>
        
        <form id="upload-form" method="post" enctype="multipart/form-data">
            <div class="upload-area" id="upload-box">
                <input type="file" name="file" id="file-input" accept="image/*">
                <label for="file-input">Click here to select an image</label>
            </div>
            <img id="preview" src="#" alt="Image Preview"/>
            <button type="submit" class="btn" id="submit-btn">Diagnose</button>
        </form>
        
        <div class="loader" id="loader"></div>

        {% if result %}
        <div class="results-card">
            <h2>🔬 Analysis Results</h2>
            <div class="result-item">
                <span>Prediction:</span>
                <span>{{ result.prediction }}</span>
            </div>
            <div class="result-item">
                <span>Confidence:</span>
                <span>{{ result.confidence }}%</span>
            </div>
            
            {% if result.remedy %}
            <div class="remedy">
                <h3>Organic Remedy</h3>
                <p>{{ result.remedy.organic }}</p>
                <h3>Chemical Remedy</h3>
                <p>{{ result.remedy.chemical }}</p>
            </div>
            {% else %}
            <div class="remedy">
                <h3>✅ Healthy Plant</h3>
                <p>Great news! This plant appears to be healthy. Continue with good care practices.</p>
            </div>
            {% endif %}
        </div>
        <img src="data:image/jpeg;base64,{{ result.image }}" style="max-width:100%; margin-top:20px; border-radius:8px;" alt="Uploaded Image">
        {% endif %}
    </div>

    <script>
        const fileInput = document.getElementById('file-input');
        const preview = document.getElementById('preview');
        const submitBtn = document.getElementById('submit-btn');
        const uploadForm = document.getElementById('upload-form');
        const loader = document.getElementById('loader');

        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    submitBtn.style.display = 'inline-block';
                }
                reader.readAsDataURL(file);
            }
        });

        uploadForm.addEventListener('submit', function() {
            submitBtn.style.display = 'none';
            loader.style.display = 'block';
        });
    </script>
</body>
</html>
"""

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template_string(HTML_TEMPLATE, error="No file selected")
        
        file = request.files['file']
        if file.filename == '':
            return render_template_string(HTML_TEMPLATE, error="No file selected")

        if file and model:
            image_bytes = file.read()
            processed_image = preprocess_image(image_bytes)
            
            prediction = model.predict(processed_image)
            class_index = np.argmax(prediction)
            class_name = CLASS_NAMES[class_index]
            confidence = round(np.max(prediction) * 100, 2)

            is_healthy = 'healthy' in class_name
            remedy = REMEDY_SUGGESTIONS.get(class_name, REMEDY_SUGGESTIONS['default']) if not is_healthy else None

            result_data = {
                "prediction": class_name.replace('', ' - ').replace('', ' '),
                "confidence": confidence,
                "remedy": remedy,
                "image": base64.b64encode(image_bytes).decode('utf-8')
            }
            return render_template_string(HTML_TEMPLATE, result=result_data)

    return render_template_string(HTML_TEMPLATE)

# --- Main execution ---
if __name__ == '_main_':
    load_plant_model()
    if model is None:
        print("Model could not be loaded. The application will not work correctly.")
    app.run(debug=True, port=5001)