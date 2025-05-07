import os
import base64
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import json
from dataset import load_preprocessed_fer2013, create_data_loaders
from flask import Flask, render_template, request, jsonify
from dataset_training import EmotionCNN, train_model, METRICS_PATH
from predict import predict_emotion

app = Flask(__name__)
os.makedirs('static/uploads', exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
            return jsonify({'success': True, 'metrics': metrics})
        else:
            return jsonify({'success': False, 'error': 'No metrics available. Train the model first.'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Loading dataset...")
        train_dataset, val_dataset, test_dataset, _ = load_preprocessed_fer2013()
        
        # Use optimal batch size for CNN training
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=64
        )
        
        print("Initializing model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EmotionCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        
        # Updated learning rate to optimal value
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print("Starting training...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,  # Add test_loader for metrics calculation
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=25
        )
        
        return jsonify({
            'success': True,
            'message': 'Training completed successfully',
            'training_history': {
                'train_losses': [float(loss) for loss in history['train_losses']],
                'val_losses': [float(loss) for loss in history['val_losses']],
                'train_accuracies': [float(acc) for acc in history['train_accuracies']],
                'val_accuracies': [float(acc) for acc in history['val_accuracies']]
            }
        })
    except Exception as e:
        import traceback
        print("Training error:", str(e))
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if not request.files.get('image'):
        return jsonify({'error': 'No image uploaded'}), 400
    
    # Check if model exists
    model_path = os.path.join('model', 'emotion_model.pth')
    if not os.path.exists(model_path):
        return jsonify({'success': False, 'error': 'Please train the model first before uploading an image'}), 400
    
    try:
        image_bytes = request.files['image'].read()
        result = predict_emotion(image_bytes)
        
        if result['success']:
            _, buffer = cv2.imencode('.jpg', result['face_image'])
            return jsonify({
                'success': True,
                'emotion': result['emotion'],
                'confidence': float(result['probability'] * 100),
                'probabilities': result['probabilities'],
                'face_image': base64.b64encode(buffer).decode('utf-8')
            })
        return jsonify({'error': 'No face detected in the image'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)