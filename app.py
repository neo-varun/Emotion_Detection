from flask import Flask, render_template, request, jsonify
import os
from predict import predict_emotion
from dataset_training import train_model_with_callback, EmotionCNN
import base64
import cv2
import numpy as np

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image file
        image_bytes = file.read()
        
        # Get prediction
        result = predict_emotion(image_bytes)
        
        if result['success']:
            # Convert face image to base64 for display
            _, buffer = cv2.imencode('.jpg', result['face_image'])
            face_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'emotion': result['emotion'],
                'confidence': float(result['probability'] * 100),
                'probabilities': result['probabilities'],
                'face_image': face_image_b64
            })
        else:
            return jsonify({'error': 'No face detected in the image'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        # Start training using preprocessed dataset
        history = train_model_with_callback()
        
        return jsonify({
            'success': True,
            'message': 'Training completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)