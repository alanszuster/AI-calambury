from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import numpy as np
import json
import base64
from PIL import Image
import io
import tensorflow as tf
import os
from functools import wraps
from model.drawing_classifier import DrawingClassifier

app = Flask(__name__)

# Configure CORS for specific domains
allowed_origins = [
    # Production URLs dla alanszuster.page
    "https://alanszuster.vercel.app",
    "https://alanszusterpage-alanszuster-alanszusters-projects.vercel.app",
    "https://alanszusterpage-alanszusters-projects.vercel.app",
    "https://alanszuster.github.io",

    # Development URLs
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://localhost:8000"
]

if os.getenv('FLASK_ENV') == 'development':
    allowed_origins.extend([
        "http://localhost:5000",
        "http://127.0.0.1:3000"
    ])

CORS(app, origins=allowed_origins)

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

classifier = None

# Simple API key authentication
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        expected_key = os.getenv('API_KEY', 'your-secret-api-key-here')

        # Skip API key check for health endpoint and development
        if request.endpoint == 'health_check' or os.getenv('FLASK_ENV') == 'development':
            return f(*args, **kwargs)

        if not api_key or api_key != expected_key:
            return jsonify({'error': 'Invalid or missing API key'}), 401

        return f(*args, **kwargs)
    return decorated_function

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
@limiter.limit("10 per minute")
def predict_drawing():
    try:
        # Input validation
        if not request.json or 'image' not in request.json:
            return jsonify({'error': 'Missing image data'}), 400

        data = request.json
        image_data = data['image']

        # Validate base64 image format
        if not image_data.startswith('data:image/'):
            return jsonify({'error': 'Invalid image format'}), 400

        image_data = image_data.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        if classifier is None:
            return jsonify({
                'error': 'Model not loaded',
                'predictions': []
            }), 503

        predictions = classifier.predict(image)

        return jsonify({
            'predictions': predictions,
            'success': True
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'predictions': []
        }), 500

@app.route('/classes')
@limiter.limit("30 per minute")
def get_classes():
    """Get list of supported drawing classes"""
    if classifier is None:
        return jsonify({'error': 'Model not loaded', 'classes': []}), 503

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
@limiter.limit("60 per hour")
def get_random_word():
    """Get a random word from supported classes for drawing challenges"""
    if classifier is None:
        return jsonify({'error': 'Model not loaded', 'word': None}), 503

    import random
    word = random.choice(classifier.classes)
    return jsonify({'word': word})

# Initialize model for production (Vercel)
if os.getenv('VERCEL') or os.getenv('FLASK_ENV') == 'production':
    print("Production environment detected - initializing model...")
    init_model()

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
