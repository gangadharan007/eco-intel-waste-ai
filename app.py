import os, io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model/waste_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# IMPORTANT: must match training class index order
CLASS_NAMES = ["hazardous", "organic", "recyclable"]

ORGANIC_MINERALS = ["N","P","K","Ca","Mg","S","Fe","Zn","Mn","Cu"]

def preprocess_image(file_storage):
    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    img = img.resize((224, 224))
    x = np.array(img).astype("float32") / 255.0
    return np.expand_dims(x, axis=0)

@app.get("/")
def health():
    return jsonify({"status": "ok", "classes": CLASS_NAMES})

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    f = request.files["image"]
    x = preprocess_image(f)

    probs = model.predict(x, verbose=0)[0]   # (3,)
    idx = int(np.argmax(probs))
    waste_type = CLASS_NAMES[idx]
    confidence = round(float(probs[idx]) * 100, 2)

    return jsonify({
        "predicted_waste_type": waste_type,
        "confidence": confidence,
        "recyclable": waste_type == "recyclable",
        "minerals": ORGANIC_MINERALS if waste_type == "organic" else None
    })
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
