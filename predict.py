import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from dataset_training import EmotionCNN
from data_preprocessing import crop_face_mediapipe
from torch.amp import autocast

EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

def load_model():
    model_path = os.path.join('model', 'emotion_model.pth')
    if not os.path.exists(model_path):
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    
    return model, device

def preprocess_image(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    face_image = crop_face_mediapipe(image)
    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    tensor = transform(Image.fromarray(gray_image)).unsqueeze(0)
    return tensor, face_image

def predict_emotion(image_bytes):
    try:
        model, device = load_model()
        if model is None:
            return {'success': False, 'error': 'Please train the model first to make predictions'}

        tensor, face_image = preprocess_image(image_bytes)
        tensor = tensor.to(device, non_blocking=True)
        
        with torch.no_grad(), autocast(device_type=device.type, enabled=device.type=='cuda'):
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probs).item()
        
        # Move results to CPU for numpy operations
        probs = probs.cpu()
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'success': True,
            'emotion': EMOTION_LABELS[predicted_class],
            'probability': probs[predicted_class].item(),
            'probabilities': {EMOTION_LABELS[i]: p.item() for i, p in enumerate(probs)},
            'face_image': face_image
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}