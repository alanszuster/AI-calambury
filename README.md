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
- Python 3.8+
- TensorFlow 2.x
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

## 🚀 Quick Start

### Option 1: Use Pre-trained Model
```bash
# Just start the API (uses existing model)
python app.py
```

### Option 2: Train Your Own Model
```bash
# 1. Prepare dataset (~10-20 minutes, ~2GB download)
python scripts/prepare_data.py

# 2. Train model (~30-60 minutes)
python scripts/train_model.py

# 3. Start API
python app.py
```

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

## 🧪 Model Training & Improvement

See [`docs/TRAINING.md`](docs/TRAINING.md) for detailed instructions on:

- Data preparation and augmentation
- Model architecture customization
- Training hyperparameter tuning
- Performance evaluation and optimization
- Adding new drawing classes

## 📁 Project Structure

```text
ai-calambury/
├── app.py                    # 🚀 Main Flask API server
├── config.py                 # ⚙️  Central configuration
├── requirements.txt          # 📦 Python dependencies
├── runtime.txt              # 🐍 Python version for deployment
├── Procfile                 # 🚀 Heroku deployment config
├── README.md                # 📖 Main documentation
├── docs/                    # 📚 Documentation
│   ├── API.md              # 🔌 API documentation
│   ├── TRAINING.md         # 🎯 Training guide
│   ├── DEPLOYMENT.md       # 🚀 Deployment instructions
│   └── README.md           # 📖 Docs overview
├── scripts/                 # 🛠️  Data & training scripts
│   ├── prepare_data.py     # 📥 Data download & preprocessing
│   ├── train_model.py      # 🎯 Model training
│   └── README.md           # 📖 Scripts documentation
├── docker/                  # 🐳 Docker configuration
│   ├── Dockerfile          # 🐳 Container definition
│   ├── .dockerignore       # 🚫 Docker ignore rules
│   └── README.md           # 📖 Docker documentation
├── dataset/                 # 📊 All data files
│   ├── quickdraw/          # 📁 Raw Quick Draw! data
│   ├── processed/          # ⚡ Preprocessed training data
│   └── README.md           # 📖 Dataset documentation
├── model/                   # 🧠 AI model files
│   ├── drawing_classifier.py  # 🤖 Classifier implementation
│   ├── drawing_model.h5       # 💾 Trained model weights
│   └── classes.json           # 🏷️  Class labels mapping
├── outputs/                 # 📈 Generated outputs
│   ├── training_history.png   # 📊 Training plots
│   └── README.md              # 📖 Outputs documentation
├── logs/                    # 📝 Training & error logs
│   └── README.md            # 📖 Logs documentation
└── experiments/             # 🔬 Research experiments
    └── README.md            # 📖 Experiments documentation
```

## 🚀 Deployment

### Docker
```bash
# Build image
docker build -f docker/Dockerfile -t ai-drawing-classifier .

# Run container
docker run -p 5000:5000 ai-drawing-classifier
```

### Heroku
```bash
# Deploy to Heroku
git push heroku main
```

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for detailed deployment instructions.

## 🧮 Technical Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 28x28 grayscale images
- **Dataset**: Google Quick Draw! (simplified drawings)
- **Framework**: TensorFlow/Keras
- **Preprocessing**: PIL/Pillow image processing
- **API**: Flask with CORS support

## 📊 Performance

- **Training Accuracy**: ~85-90%
- **Validation Accuracy**: ~80-85%
- **Inference Time**: <100ms per prediction
- **Model Size**: ~2MB

## 🔮 Future Improvements

- [ ] Support for more drawing classes
- [ ] Data augmentation techniques
- [ ] Model architecture optimization
- [ ] Batch prediction support
- [ ] Real-time streaming prediction
- [ ] Model versioning and A/B testing

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
