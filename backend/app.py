import os
import sys
import base64
import io
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

import numpy as np
import onnxruntime as ort

app = Flask(__name__)
CORS(app) # Enable CORS for frontend domain requests

# Load ONNX ResNet101 model globally on startup (search backend/ or parent/ or model/)
onnx_model_path = Path(__file__).resolve().parent / 'final_model.onnx'
if not onnx_model_path.exists():
    onnx_model_path = Path(__file__).resolve().parent.parent / 'final_model.onnx'
if not onnx_model_path.exists():
    onnx_model_path = Path(__file__).resolve().parent / 'model' / 'final_model.onnx'
if not onnx_model_path.exists():
    raise RuntimeError(f"ONNX model weights not found at: {onnx_model_path}")
print(f"Loading ONNX ResNet18 model from: {onnx_model_path}")

# Initialize ONNX inference session on CPU for minimal memory footprint
session = ort.InferenceSession(str(onnx_model_path), providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ResNet preprocessing implemented in NumPy/Pillow
def preprocess_image(pil_img):
    pil_img = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    img_onnx = img_np.transpose((2, 0, 1))
    img_onnx = np.expand_dims(img_onnx, axis=0)
    return img_onnx

@app.route('/')
def index():
    return "Spam Investigator API is running! Made by Daksh Arora"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Parse base64 image data
        image_data_b64 = data['image']
        if ',' in image_data_b64:
            # Strip header if present
            image_data_b64 = image_data_b64.split(',')[1]
            
        image_bytes = base64.b64decode(image_data_b64)
        image_file = io.BytesIO(image_bytes)
        
        # Load image as PIL
        pil_image = Image.open(image_file).convert('RGB')
        img_onnx = preprocess_image(pil_image)
        
        logits = session.run([output_name], {input_name: img_onnx})[0]
        
        # Softmax function
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e_x / e_x.sum(axis=1, keepdims=True)
        probs = probs[0]
        
        # Class 1 is Screen Recapture (Spoof)
        prob = float(probs[1])
        
        return jsonify({
            'probability': float(prob),
            'device': 'CPU (ONNX Runtime)'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
