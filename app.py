from flask import Flask, render_template, request, jsonify
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
    return render_template('index.html')

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

@app.route('/get_random_word')
def get_random_word():
    words = [
        'cat', 'dog', 'house', 'car', 'flower', 'tree', 'sun', 'moon',
        'fish', 'bird', 'apple', 'chair', 'table', 'book', 'phone', 'computer',
        'guitar', 'ball', 'cake', 'umbrella', 'bicycle', 'bus', 'airplane', 'boat'
    ]
    import random
    word = random.choice(words)
    return jsonify({'word': word})

if __name__ == '__main__':
    print("Starting AI Pictionary...")
    print("Initializing AI model...")
    init_model()

    print("Server will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
