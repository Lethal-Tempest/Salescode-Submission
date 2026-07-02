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
        import torch
        import torchvision
        from torchvision import transforms
        from PIL import Image
    except ImportError:
        print("Error: PyTorch, torchvision, and Pillow are required to run this script.")
        sys.exit(1)
        
    # Device initialization with robust Blackwell architecture check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        try:
            # Force kernel execution to check sm_120 compatibility
            (torch.zeros(1).to(device) + 1).cpu()
        except Exception:
            device = torch.device('cpu')
            
    # Load ResNet101 model weights
    resnet_model_path = Path(__file__).resolve().parent / 'final_model.pth'
    if not resnet_model_path.exists():
        resnet_model_path = Path(__file__).resolve().parent.parent / 'final_model.pth'
    if not resnet_model_path.exists():
        resnet_model_path = Path(__file__).resolve().parent / 'model' / 'final_model.pth'
    if not resnet_model_path.exists():
        print(f"Error: Model weights not found at: {resnet_model_path}")
        sys.exit(1)
        
    try:
        model = torchvision.models.resnet101()
        model.fc = torch.nn.Linear(2048, 3)
        model.load_state_dict(torch.load(str(resnet_model_path), map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        img = Image.open(image_path).convert('RGB')
        tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        # Class 2 is Screen Recapture (Spoof)
        prob = float(probs[2])
        print(f"{prob:.2f}")
    except Exception as e:
        print(f"Error running inference: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
