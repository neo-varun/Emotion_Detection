import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_preprocessing import modify_dataset_transforms, preprocess_dataset_with_mediapipe
from torchvision.datasets import ImageFolder
import multiprocessing

# Set multiprocessing start method to 'spawn' on Windows
if os.name == 'nt':  # Windows
    multiprocessing.set_start_method('spawn', force=True)

def download_fer2013(use_mediapipe=False):
    data_dir = os.path.join(os.getcwd(), 'data', 'fer2013')
    
    print("Loading FER-2013 dataset from folders...")
    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),  # FER standard size
        transforms.ToTensor(),
    ])
    
    # Load datasets from folders
    try:
        train_dataset = ImageFolder(
            root=os.path.join(data_dir, 'train'),
            transform=transform
        )
        
        val_dataset = ImageFolder(
            root=os.path.join(data_dir, 'val'),
            transform=transform
        )
        
        test_dataset = ImageFolder(
            root=os.path.join(data_dir, 'test'),
            transform=transform
        )
        
        print("Dataset loaded successfully!")
        print(f"Train: {len(train_dataset)} images")
        print(f"Validation: {len(val_dataset)} images")
        print(f"Test: {len(test_dataset)} images")
        
        # Apply MediaPipe face detection and cropping if requested
        if use_mediapipe:
            print("Applying MediaPipe face detection and cropping...")
            train_dataset = modify_dataset_transforms(train_dataset)
            val_dataset = modify_dataset_transforms(val_dataset)
            test_dataset = modify_dataset_transforms(test_dataset)
            print("MediaPipe processing applied to all datasets")
        
        emotion_labels = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        return train_dataset, val_dataset, test_dataset, emotion_labels
        
    except Exception as e:
        raise RuntimeError(
            f"Error loading dataset from folders: {str(e)}\n"
            "Please ensure the dataset is organized in the following structure:\n"
            "data/fer2013/\n"
            "  ├── train/\n"
            "  │   ├── 0/\n"
            "  │   ├── 1/\n"
            "  │   └── ...\n"
            "  ├── val/\n"
            "  │   ├── 0/\n"
            "  │   ├── 1/\n"
            "  │   └── ...\n"
            "  └── test/\n"
            "      ├── 0/\n"
            "      ├── 1/\n"
            "      └── ..."
        ) from e

def load_preprocessed_fer2013():
    """Load the preprocessed FER2013 dataset"""
    data_dir = os.path.join(os.getcwd(), 'data', 'fer2013')
    preprocessed_dir = os.path.join(os.getcwd(), 'data', 'fer2013_mediapipe')
    # Check if preprocessed data exists, if not create it
    if not os.path.exists(preprocessed_dir):
        print("Preprocessed dataset not found. Creating it now (this may take a while)...")
        preprocess_dataset_with_mediapipe(data_dir, preprocessed_dir)
    
    # Define transform that explicitly converts to grayscale
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale output
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load preprocessed datasets
    try:
        train_dataset = ImageFolder(
            root=os.path.join(preprocessed_dir, 'train'),
            transform=transform
        )
        
        val_dataset = ImageFolder(
            root=os.path.join(preprocessed_dir, 'val'),
            transform=transform
        )
        
        test_dataset = ImageFolder(
            root=os.path.join(preprocessed_dir, 'test'),
            transform=transform
        )
        
        print("Preprocessed dataset loaded successfully!")
        print(f"Train: {len(train_dataset)} images")
        print(f"Validation: {len(val_dataset)} images")
        print(f"Test: {len(test_dataset)} images")
        
        emotion_labels = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        return train_dataset, val_dataset, test_dataset, emotion_labels
        
    except Exception as e:
        raise RuntimeError(
            f"Error loading preprocessed dataset: {str(e)}"
        ) from e

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=64):
    # Determine if we're using MediaPipe by checking the transform type
    using_mediapipe = hasattr(train_dataset, 'transform') and 'mediapipe_transform' in str(train_dataset.transform)
    
    # Set num_workers to 0 if using MediaPipe on Windows or if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0 if (using_mediapipe or device.type == 'cuda' or os.name == 'nt') else 4
    
    # Use persistent workers to avoid recreation of worker processes
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=device.type == 'cuda'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=device.type == 'cuda'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=device.type == 'cuda'
    )
    
    return train_loader, val_loader, test_loader