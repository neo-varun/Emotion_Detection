import cv2
import mediapipe as mp
import os
from tqdm import tqdm
import shutil
import warnings
import logging

# Suppress warnings and logging
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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

        results = self.detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.detections:
            return cv2.resize(image, (48, 48))
        
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w = image.shape[:2]
        
        # Calculate coordinates
        xmin = max(0, int(bbox.xmin * w))
        ymin = max(0, int(bbox.ymin * h))
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Expand box
        expand_x = int(width * expand_ratio)
        expand_y = int(height * expand_ratio)
        xmin = max(0, xmin - expand_x)
        ymin = max(0, ymin - expand_y)
        xmax = min(w, xmin + width + 2 * expand_x)
        ymax = min(h, ymin + height + 2 * expand_y)
        
        # Crop and resize
        return cv2.resize(image[ymin:ymax, xmin:xmax], (48, 48))

face_detector = FaceDetector()

def crop_face_mediapipe(image, expand_ratio=0.1):

    return face_detector.detect(image, expand_ratio)

def preprocess_dataset_with_mediapipe(data_dir, output_dir):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        output_split_dir = os.path.join(output_dir, split)
        
        for emotion_class in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, emotion_class)
            output_class_dir = os.path.join(output_split_dir, emotion_class)
            os.makedirs(output_class_dir, exist_ok=True)
            
            image_files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Processing {split}/{emotion_class}: {len(image_files)} images")
            
            for img_file in tqdm(image_files, desc=f"{split}/{emotion_class}"):
                try:
                    img_path = os.path.join(class_dir, img_file)
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                        
                    face_image = crop_face_mediapipe(image)
                    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join(output_class_dir, img_file), gray_image)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
    
    print("Dataset preprocessing completed!")