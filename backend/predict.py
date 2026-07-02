import sys
import os
from pathlib import Path

# Suppress PyTorch/CUDA and other logging to stdout for clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        sys.exit(1)
        
    # Lazy imports to minimize CLI startup overhead
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except ImportError:
        print("Error: onnxruntime, numpy, and Pillow are required to run this script.")
        sys.exit(1)
        
    # Load ONNX ResNet18 model weights
    onnx_model_path = Path(__file__).resolve().parent / 'final_model.onnx'
    if not onnx_model_path.exists():
        onnx_model_path = Path(__file__).resolve().parent.parent / 'final_model.onnx'
    if not onnx_model_path.exists():
        onnx_model_path = Path(__file__).resolve().parent / 'model' / 'final_model.onnx'
    if not onnx_model_path.exists():
        print(f"Error: Model weights not found at: {onnx_model_path}")
        sys.exit(1)
        
    try:
        session = ort.InferenceSession(str(onnx_model_path), providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Preprocessing matching PyTorch transforms
        img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        
        img_onnx = img_np.transpose((2, 0, 1))
        img_onnx = np.expand_dims(img_onnx, axis=0)
        
        logits = session.run([output_name], {input_name: img_onnx})[0]
        
        # Softmax function
        e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = e_x / e_x.sum(axis=1, keepdims=True)
        probs = probs[0]
        
        # Class 1 is Screen Recapture (Spoof)
        prob = float(probs[1])
        print(f"{prob:.2f}")
    except Exception as e:
        print(f"Error running inference: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
