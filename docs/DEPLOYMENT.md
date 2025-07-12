# Deployment Instructions for AI Drawing Classifier

## Local Development Setup

### 1. Environment Setup
```bash
# Clone repository
git clone <your-repo-url>
cd ai-calambury

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Start (with demo model)
```bash
python setup.py  # Creates basic model
python app.py    # Starts application
```

### 3. Full Setup (with model training)
```bash
python prepare_data.py  # Downloads Quick Draw! data (~1GB)
python train_model.py   # Trains model (~30 minutes)
python app.py          # Starts application
```

## Heroku Deployment

### 1. Prepare Files
Create `Procfile`:
```
web: python app.py
```

Create `runtime.txt`:
```
python-3.9.18
```

### 2. Heroku Configuration
```bash
# Login to Heroku
heroku login

# Create application
heroku create ai-drawing-classifier-demo

# Set environment variables
heroku config:set FLASK_ENV=production

# Deploy
git add .
git commit -m "Initial deployment"
git push heroku main
```

### 3. Heroku Notes
- Heroku has memory limits, so use a smaller model
- Add `.slugignore` to avoid uploading large data files
- Use CPU-optimized TensorFlow

## Vercel Deployment

### 1. Vercel Structure
Create `vercel.json`:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

### 2. Deploy
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

## DigitalOcean App Platform Deployment

### 1. Konfiguracja
Stwórz `.do/app.yaml`:
```yaml
name: ai-kalambury
services:
- name: web
  source_dir: /
  github:
    repo: your-username/ai-calambury
    branch: main
  run_command: python app.py
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  http_port: 5000
  routes:
  - path: /
```

## AWS EC2 Deployment

### 1. Setup serwera
```bash
# On EC2 server
sudo apt update
sudo apt install python3 python3-pip nginx

# Klonuj repo
git clone <your-repo>
cd ai-calambury

# Instaluj zależności
pip3 install -r requirements.txt

# Uruchom z Gunicorn
pip3 install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 2. Konfiguracja Nginx
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Production Optimization

### 1. Zmniejszenie rozmiaru modelu
```python
# W train_model.py, użyj model quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### 2. Caching predykcji
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(image_hash):
    # Cache results for identical images
    pass
```

### 3. Asynchroniczne przetwarzanie
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_predict(image):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, model.predict, image)
    return result
```

## Monitoring i logi

### 1. Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
```

### 2. Health check endpoint
```python
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'model_loaded': classifier is not None}
```

## Security

### 1. Rate limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict_drawing():
    # ...
```

### 2. Input validation
```python
def validate_image(image_data):
    # Check size, format, etc.
    if len(image_data) > MAX_IMAGE_SIZE:
        raise ValueError("Image too large")
    # ...
```

## Troubleshooting

### Common issues:
1. **Out of memory** - Reduce batch_size or use smaller model
2. **Slow predictions** - Use TensorFlow Lite or model quantization
3. **CORS errors** - Make sure flask-cors is properly configured
4. **Model not loading** - Check model file paths
