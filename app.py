import os, io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Use absolute path so it works in Railway/Render regardless of cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "waste_classifier.h5")

# Loads a model saved via model.save() / .h5. [web:1268]
model = tf.keras.models.load_model(MODEL_PATH)

# IMPORTANT: must match training class index order
CLASS_NAMES = ["hazardous", "organic", "recyclable"]
ORGANIC_MINERALS = ["N", "P", "K", "Ca", "Mg", "S", "Fe", "Zn", "Mn", "Cu"]

def preprocess_image(file_storage):
    # Ensure stream is at start (important when forwarding/proxying)
    try:
        file_storage.stream.seek(0)
    except Exception:
        pass

    # Flask uploads are accessed via request.files and are file-like. [web:1290]
    img_bytes = file_storage.read()
    if not img_bytes:
        raise ValueError("Empty file")

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    x = np.array(img).astype("float32") / 255.0
    return np.expand_dims(x, axis=0)

@app.get("/")
def health():
    return jsonify({"status": "ok", "classes": CLASS_NAMES, "model_path": MODEL_PATH})

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Use multipart/form-data with key 'image'."}), 400

    f = request.files["image"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        x = preprocess_image(f)
    except Exception as e:
        return jsonify({"error": "Invalid image", "detail": str(e)}), 400

    probs = model.predict(x, verbose=0)[0]
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
    # keep debug False in production
    app.run(host="0.0.0.0", port=port, debug=False)
