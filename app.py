from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# Load model
model = load_model("potato_disease_model.h5")

# Define class names (update these with your actual classes)
class_names = ["Early Blight", "Late Blight", "Healthy"]

# Initialize Flask app
app = Flask(__name__)

# Image preprocessing function
def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        img_bytes = file.read()
        processed = preprocess_image(img_bytes)
        prediction = model.predict(processed)
        class_index = np.argmax(prediction[0])
        result = class_names[class_index]
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Potato Disease Detection API is running!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
