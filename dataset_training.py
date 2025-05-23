import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable CUDA optimizations
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

MODEL_PATH = os.path.join('model', 'emotion_model.pth')
METRICS_PATH = os.path.join('model', 'metrics.json')
os.makedirs('model', exist_ok=True)

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        
        # Initialize layers with CUDA optimized memory format
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1).to(memory_format=torch.channels_last if device.type == 'cuda' else torch.contiguous_format)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1).to(memory_format=torch.channels_last if device.type == 'cuda' else torch.contiguous_format)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(memory_format=torch.channels_last if device.type == 'cuda' else torch.contiguous_format)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1).to(memory_format=torch.channels_last if device.type == 'cuda' else torch.contiguous_format)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 12 * 12, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 7)
    
    def forward(self, x):
        # Convert to channels last format if using CUDA
        if device.type == 'cuda':
            x = x.to(memory_format=torch.channels_last)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn5(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

def calculate_metrics(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad(), autocast(device_type=device.type, enabled=device.type=='cuda'):
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Calculate per-emotion metrics
    per_emotion_metrics = {}
    emotion_labels = {
        0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
        4: 'Sad', 5: 'Surprise', 6: 'Neutral'
    }
    
    for emotion_idx in range(7):
        mask = all_labels == emotion_idx
        if np.any(mask):
            emotion_preds = all_preds[mask]
            emotion_labels_masked = all_labels[mask]
            emotion_accuracy = accuracy_score(emotion_labels_masked, emotion_preds)
            # Changed average parameter from 'binary' to 'weighted' for multiclass
            emotion_precision, emotion_recall, emotion_f1, _ = precision_recall_fscore_support(
                emotion_labels_masked, emotion_preds, average='weighted'
            )
            
            per_emotion_metrics[emotion_labels[emotion_idx]] = {
                'accuracy': float(emotion_accuracy),
                'precision': float(emotion_precision),
                'recall': float(emotion_recall),
                'f1_score': float(emotion_f1)
            }
    
    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score)
        },
        'per_emotion': per_emotion_metrics
    }
    
    # Save metrics to file
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def train_model(model, train_loader, val_loader, test_loader=None, criterion=None, optimizer=None, num_epochs=20):
    model.to(device)
    best_val_accuracy = 0.0
    history = {
        'train_losses': [], 'val_losses': [],
        'train_accuracies': [], 'val_accuracies': []
    }
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = correct = total = 0
        
        # Training phase
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training phase:")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Use mixed precision training if CUDA is available
            if device.type == 'cuda':
                with autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100.0 * correct/total:.2f}%")
            
            # Clear cache periodically
            if device.type == 'cuda' and (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total
        history['train_losses'].append(train_loss)
        history['train_accuracies'].append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = correct = total = 0
        print("\nValidation phase:")
        
        with torch.no_grad(), autocast(device_type=device.type, enabled=device.type=='cuda'):
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        history['val_losses'].append(val_loss)
        history['val_accuracies'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"\nModel saved with validation accuracy: {best_val_accuracy*100:.2f}%")
        
        # Clear cache after validation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print("\nTraining completed!")
    
    # Calculate and save metrics if test_loader is provided
    if test_loader is not None:
        print("\nCalculating model metrics...")
        metrics = calculate_metrics(model, test_loader)
        print("\nMetrics calculation completed and saved!")
    
    return history