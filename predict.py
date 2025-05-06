import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from dataset_training import EmotionCNN
from data_preprocessing import crop_face_mediapipe
import os

# Define emotion labels
emotion_labels = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

def load_model():
    """Load the trained emotion detection model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN()
    model_path = os.path.join('model', 'best_emotion_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_bytes):
    """Preprocess the image bytes for prediction"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect and crop face
    face_image = crop_face_mediapipe(image)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(gray_image)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Apply transforms and add batch dimension
    tensor = transform(pil_image).unsqueeze(0)
    return tensor, face_image

def predict_emotion(image_bytes):
    """
    Predict emotion from image bytes
    
    Returns:
        dict: Contains predicted emotion, probabilities dict, and processed face image
    """
    try:
        # Load model
        model, device = load_model()
        
        # Preprocess image
        tensor, face_image = preprocess_image(image_bytes)
        
        # Make prediction
        tensor = tensor.to(device)
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
        
        # Get prediction results
        predicted_emotion = emotion_labels[predicted_class]
        probs_dict = {emotion_labels[i]: prob.item() for i, prob in enumerate(probabilities)}
        
        return {
            'success': True,
            'emotion': predicted_emotion,
            'probability': probabilities[predicted_class].item(),
            'probabilities': probs_dict,
            'face_image': face_image
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }