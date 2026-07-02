import os
import sys
import base64
import io
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from PIL import Image

import torch
import torchvision
from torchvision import transforms

app = Flask(__name__)

# Device initialization with robust Blackwell architecture test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

if device.type == 'cuda':
    try:
        # Force kernel execution to check sm_120 compatibility
        (torch.zeros(1).to(device) + 1).cpu()
        print(f"PyTorch using device: {device} ({device_name})")
    except Exception as e:
        print(f"CUDA initialization failed ({e}). Falling back to CPU.")
        device = torch.device('cpu')
        device_name = "CPU (CUDA Incompatible)"

# Load ResNet18 model globally on startup
resnet_model_path = Path(__file__).resolve().parent / 'model' / 'final_model.pth'
if not resnet_model_path.exists():
    raise RuntimeError(f"ResNet18 model weights not found at: {resnet_model_path}")
print(f"Loading ResNet18 model from: {resnet_model_path}")

resnet_model = torchvision.models.resnet18()
resnet_model.fc = torch.nn.Linear(512, 2)
resnet_model.load_state_dict(torch.load(str(resnet_model_path), map_location=device))
resnet_model.to(device)
resnet_model.eval()

# ResNet preprocessing transforms
resnet_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

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
        tensor = resnet_transforms(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = resnet_model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        # Class 1 is Screen Recapture (Spoof)
        prob = float(probs[1])
        active_device = f"CUDA ({device_name})" if device.type == 'cuda' else "CPU"
        
        return jsonify({
            'probability': float(prob),
            'device': active_device
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
