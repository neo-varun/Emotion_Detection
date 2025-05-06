import cv2
import numpy as np
import mediapipe as mp
import os
import torch
from torchvision import transforms
from PIL import Image
import warnings
from tqdm import tqdm
import shutil
import logging

# Comprehensive warning and logging suppression
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

class FaceDetector:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceDetector, cls).__new__(cls)
            cls._instance.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
        return cls._instance
    
    def detect(self, image, expand_ratio=0.1):
        """
        Detect and crop face from image
        
        Args:
            image: Input image (numpy array, BGR format from OpenCV)
            expand_ratio: Ratio to expand the detected face bounding box
            
        Returns:
            Cropped face image
        """
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe Face Detection
        results = self.detector.process(image_rgb)
        
        # If no face is detected, return the original image
        if not results.detections:
            return cv2.resize(image, (48, 48))
        
        # Use the first detected face (closest/largest face)
        detection = results.detections[0]
        
        # Get bounding box coordinates
        bbox = detection.location_data.relative_bounding_box
        h, w, c = image.shape
        
        # Calculate absolute coordinates
        xmin = max(0, int(bbox.xmin * w))
        ymin = max(0, int(bbox.ymin * h))
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Expand the bounding box
        expand_x = int(width * expand_ratio)
        expand_y = int(height * expand_ratio)
        
        # Ensure the expanded box is within image boundaries
        xmin = max(0, xmin - expand_x)
        ymin = max(0, ymin - expand_y)
        xmax = min(w, xmin + width + 2 * expand_x)
        ymax = min(h, ymin + height + 2 * expand_y)
        
        # Crop the face
        face_image = image[ymin:ymax, xmin:xmax]
        
        # Resize to 48x48 (to match FER2013 dimensions)
        face_image = cv2.resize(face_image, (48, 48))
        
        return face_image

# Create a global face detector instance
face_detector = FaceDetector()

def crop_face_mediapipe(image, expand_ratio=0.1):
    """
    Crop the face from an image using MediaPipe Face Detection.
    
    Args:
        image: Input image (numpy array, BGR format from OpenCV)
        expand_ratio: Ratio to expand the detected face bounding box
    
    Returns:
        Cropped face image or original image if no face is detected
    """
    return face_detector.detect(image, expand_ratio)

def preprocess_image_for_emotion(image_path, grayscale=True):
    """
    Preprocess an image for emotion detection:
    1. Load the image
    2. Detect and crop the face
    3. Convert to grayscale (optional)
    4. Resize to 48x48 (FER2013 format)
    5. Convert to PyTorch tensor
    
    Args:
        image_path: Path to the input image
        grayscale: Whether to convert the image to grayscale
    
    Returns:
        Preprocessed image as a PyTorch tensor
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Crop face
    face_image = crop_face_mediapipe(image)
    
    # Convert to grayscale if needed
    if grayscale:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Convert to PIL Image
    if grayscale:
        pil_image = Image.fromarray(face_image)
    else:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_image)
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) if grayscale else transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    return transform(pil_image)

def preprocess_batch(image_paths, grayscale=True):
    """
    Preprocess a batch of images for emotion detection
    
    Args:
        image_paths: List of paths to images
        grayscale: Whether to convert images to grayscale
    
    Returns:
        Batch of preprocessed images as a PyTorch tensor
    """
    tensors = []
    for path in image_paths:
        try:
            tensor = preprocess_image_for_emotion(path, grayscale)
            tensors.append(tensor)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    return torch.stack(tensors) if tensors else None

class MediaPipeTransform:
    """Transform class that applies MediaPipe face detection and original transforms"""
    def __init__(self, original_transform):
        self.original_transform = original_transform
    
    def __call__(self, img):
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        
        # Apply face cropping
        cropped_face = crop_face_mediapipe(img_np)
        
        # Convert back to PIL Image
        if len(cropped_face.shape) == 2:  # Grayscale
            pil_img = Image.fromarray(cropped_face)
        else:  # Color
            pil_img = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
        
        # Apply the original transform
        return self.original_transform(pil_img)

def modify_dataset_transforms(dataset):
    """
    Modify an existing PyTorch dataset to include face cropping in its transforms
    
    Args:
        dataset: PyTorch dataset object
    
    Returns:
        Dataset with modified transforms
    """
    # Get the original transform
    original_transform = dataset.transform
    
    # Create a MediaPipeTransform instance
    dataset.transform = MediaPipeTransform(original_transform)
    
    return dataset

def preprocess_dataset_with_mediapipe(data_dir, output_dir):
    """
    Preprocess an entire dataset with MediaPipe face detection and save results
    
    Args:
        data_dir: Path to the original dataset directory
        output_dir: Path to save the preprocessed images
    """
    # Create output directory structure
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Process each split (train/val/test)
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        output_split_dir = os.path.join(output_dir, split)
        
        # Process each emotion class
        for emotion_class in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, emotion_class)
            output_class_dir = os.path.join(output_split_dir, emotion_class)
            os.makedirs(output_class_dir, exist_ok=True)
            
            # Process each image in the class
            image_files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Processing {split}/{emotion_class}: {len(image_files)} images")
            
            for img_file in tqdm(image_files, desc=f"{split}/{emotion_class}"):
                try:
                    # Read image
                    img_path = os.path.join(class_dir, img_file)
                    image = cv2.imread(img_path)
                    
                    if image is None:
                        print(f"Warning: Could not read image {img_path}")
                        continue
                    
                    # Apply face detection and cropping
                    face_image = crop_face_mediapipe(image)
                    
                    # Convert to grayscale
                    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    
                    # Save preprocessed image
                    output_path = os.path.join(output_class_dir, img_file)
                    cv2.imwrite(output_path, gray_image)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
    
    print("Dataset preprocessing completed!")