import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import multiprocessing
from data_preprocessing import preprocess_dataset_with_mediapipe

if os.name == 'nt':
    multiprocessing.set_start_method('spawn', force=True)

# Set device globally
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def load_preprocessed_fer2013():

    data_dir = os.path.join(os.getcwd(), 'data', 'fer2013')
    preprocessed_dir = os.path.join(os.getcwd(), 'data', 'fer2013_mediapipe')
    
    if not os.path.exists(preprocessed_dir):
        print("Preprocessing dataset with MediaPipe...")
        preprocess_dataset_with_mediapipe(data_dir, preprocessed_dir)
    
    # Optimize transforms for GPU memory
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    try:
        datasets = {split: ImageFolder(
            root=os.path.join(preprocessed_dir, split),
            transform=transform
        ) for split in ['train', 'val', 'test']}
        
        print("Dataset loaded successfully!")
        for split, dataset in datasets.items():
            print(f"{split.capitalize()}: {len(dataset)} images")
        
        emotion_labels = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
            4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
        return datasets['train'], datasets['val'], datasets['test'], emotion_labels
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {str(e)}")

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=64):
    """Create data loaders for training, validation and testing"""
    # Use global DEVICE
    num_workers = 0 if DEVICE.type == 'cuda' or os.name == 'nt' else 4
    
    # Optimize loader settings for CUDA
    loader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'persistent_workers': num_workers > 0,
        'pin_memory': DEVICE.type == 'cuda',
        'prefetch_factor': 2 if num_workers > 0 else None
    }
    
    return (
        DataLoader(train_dataset, shuffle=True, **loader_args),
        DataLoader(val_dataset, shuffle=False, **loader_args),
        DataLoader(test_dataset, shuffle=False, **loader_args)
    )