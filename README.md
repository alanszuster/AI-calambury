# AI Drawing Classifier API

Backend API for real-time drawing recognition using deep learning. Trained on Google's Quick Draw! dataset with support for 50 different object categories.

## 🧠 Features

- **Deep Learning CNN**: Custom architecture for drawing recognition
- **RESTful API**: Clean endpoints for prediction and metadata
- **Real-time Classification**: Fast inference on sketched drawings
- **50 Object Categories**: Wide range of recognizable objects
- **Production Ready**: Docker support and deployment configurations

## 🔬 Supported Classes

```
apple, bicycle, bird, book, car, cat, chair, circle, cloud, computer,
dog, flower, guitar, house, moon, phone, airplane, sun, table, tree,
umbrella, fish, bus, clock, coffee, cup, elephant, eye, face, fork,
hand, hat, heart, key, knife, lamp, leaf, mountain, mouse, mushroom,
pencil, pizza, rainbow, shoe, snake, star, sword, train, truck, whale
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- TensorFlow CPU 2.x
- Flask

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ai-calambury

# Install dependencies
pip install -r requirements.txt

# Start API server
python app.py
```

The API will be available at `http://localhost:5000`

**Note**: This repository contains only the production-ready API with a pre-trained model. Training scripts and datasets are available separately for development purposes.

## 📡 API Endpoints

### `GET /`
Get API information and available endpoints.

**Response:**
```json
{
  "name": "AI Drawing Classifier API",
  "version": "1.0",
  "endpoints": {
    "POST /predict": "Classify a drawing image",
    "GET /classes": "Get list of supported classes",
    "GET /health": "Check API health"
  }
}
```

### `POST /predict`
Classify a drawing image.

**Request:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Response:**
```json
{
  "predictions": [
    {"class": "cat", "confidence": 0.92},
    {"class": "dog", "confidence": 0.05},
    {"class": "bird", "confidence": 0.02}
  ],
  "success": true
}
```

### `GET /classes`
Get list of all supported drawing classes.

**Response:**
```json
{
  "classes": ["apple", "bicycle", "bird", ...],
  "total_classes": 50
}
```

### `GET /health`
Check API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "version": "1.0"
}
```

### `GET /get_random_word`
Get a random class for drawing challenges.

**Response:**
```json
{
  "word": "cat"
}
```

## 🔧 Usage Example

```python
import requests
import base64

# Read image file
with open('drawing.png', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# Make prediction
response = requests.post('http://localhost:5000/predict', json={
    'image': f'data:image/png;base64,{image_data}'
})

result = response.json()
print(f"Prediction: {result['predictions'][0]['class']}")
print(f"Confidence: {result['predictions'][0]['confidence']:.2%}")
```

## 🧪 Model Training & Development

This production repository contains only the trained model and API. For model training and development:

- Training scripts and datasets are maintained separately for development
- Contact repository maintainer for access to training resources
- Full development repository with training code available on request

## 📁 Project Structure

```text
ai-calambury/
├── app.py                    # 🚀 Main Flask API server
├── config.py                 # ⚙️  Central configuration
├── requirements.txt          # 📦 Python dependencies
├── runtime.txt              # 🐍 Python version for deployment
├── vercel.json              # 🚀 Vercel deployment config
├── Procfile                 # 🚀 Heroku deployment config
├── README.md                # 📖 Main documentation
├── model/                   # 🧠 AI model files (production-ready)
│   ├── drawing_classifier.py  # 🤖 Classifier implementation
│   ├── best_model.keras       # 💾 Pre-trained model weights
│   └── classes.json           # 🏷️  Class labels mapping
├── tests/                   # 🧪 Unit tests
│   ├── test_app.py          # 🧪 API endpoint tests
│   └── test_model.py        # 🧪 Model functionality tests
└── docs/                    # � Documentation
    ├── API.md              # � API documentation
    ├── TRAINING.md         # 🎯 Training guide
    ├── DEPLOYMENT.md       # 🚀 Deployment instructions
    └── README.md           # 📖 Docs overview
```

**Note**: Training scripts (`scripts/`), datasets (`dataset/`), and training outputs (`outputs/`) are not included in this production repository to keep it lightweight for deployment.

## 🚀 Deployment

### Vercel (Recommended)

This repository is optimized for Vercel deployment:

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### Docker

```bash
# This repository doesn't include Docker files
# Contact maintainer for Docker setup instructions
```

### Heroku

```bash
# Deploy to Heroku
git push heroku main
```

For detailed deployment instructions, contact the repository maintainer.

## 🧮 Technical Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 28x28 grayscale images
- **Dataset**: Google Quick Draw! (simplified drawings)
- **Framework**: TensorFlow CPU/Keras
- **Preprocessing**: PIL/Pillow image processing
- **API**: Flask with CORS support and rate limiting
- **Deployment**: Optimized for Vercel serverless functions

## 📊 Performance

- **Training Accuracy**: ~85-90%
- **Validation Accuracy**: ~80-85%
- **Inference Time**: <100ms per prediction
- **Model Size**: ~2MB (optimized for serverless deployment)

## 🔮 Future Improvements

- [ ] Support for more drawing classes
- [ ] TensorFlow Lite optimization for faster inference
- [ ] Batch prediction support
- [ ] Real-time streaming prediction
- [ ] Model versioning and A/B testing
- [ ] Enhanced preprocessing pipeline

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📧 Contact

For questions about model training or API usage, please open an issue.
