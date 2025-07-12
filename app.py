from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import base64
from PIL import Image
import io
import tensorflow as tf
from model.drawing_classifier import DrawingClassifier

app = Flask(__name__)
CORS(app)

classifier = None

def init_model():
    global classifier
    try:
        classifier = DrawingClassifier()
        classifier.load_model()
        print("AI model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        classifier = None

@app.route('/')
def index():
    return jsonify({
        'name': 'AI Drawing Classifier API',
        'version': '1.0',
        'endpoints': {
            'POST /predict': 'Classify a drawing image',
            'GET /classes': 'Get list of supported classes',
            'GET /health': 'Check API health'
        }
    })

@app.route('/predict', methods=['POST'])
def predict_drawing():
    try:
        data = request.json
        image_data = data['image']

        image_data = image_data.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        if classifier is None:
            return jsonify({
                'error': 'Model not loaded',
                'predictions': []
            })

        predictions = classifier.predict(image)

        return jsonify({
            'predictions': predictions,
            'success': True
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'predictions': []
        })

@app.route('/classes')
def get_classes():
    """Get list of supported drawing classes"""
    if classifier is None:
        return jsonify({'error': 'Model not loaded', 'classes': []})

    return jsonify({
        'classes': classifier.classes,
        'total_classes': len(classifier.classes)
    })

@app.route('/health')
def health_check():
    """Check API health and model status"""
    model_status = 'loaded' if classifier is not None else 'not_loaded'
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'version': '1.0'
    })

@app.route('/get_random_word')
def get_random_word():
    """Get a random word from supported classes for drawing challenges"""
    if classifier is None:
        return jsonify({'error': 'Model not loaded', 'word': None})

    import random
    word = random.choice(classifier.classes)
    return jsonify({'word': word})

if __name__ == '__main__':
    print("Starting AI Drawing Classifier API...")
    print("Initializing AI model...")
    init_model()

    print("API will be available at: http://localhost:5000")
    print("Available endpoints:")
    print("  GET  /           - API info")
    print("  POST /predict    - Classify drawing")
    print("  GET  /classes    - Get supported classes")
    print("  GET  /health     - Health check")
    print("  GET  /get_random_word - Get random class for challenges")

    app.run(debug=True, host='0.0.0.0', port=5000)
