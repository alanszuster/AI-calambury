# Instrukcje deployment dla AI Kalambury

## Wersja lokalna (development)

### 1. Przygotowanie środowiska
```bash
# Klonowanie repozytorium
git clone <your-repo-url>
cd ai-calambury

# Tworzenie wirtualnego środowiska (opcjonalne ale zalecane)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# lub
venv\Scripts\activate     # Windows

# Instalacja zależności
pip install -r requirements.txt
```

### 2. Szybki start (z modelem demonstracyjnym)
```bash
python setup.py  # Tworzy podstawowy model
python app.py    # Uruchamia aplikację
```

### 3. Pełny setup (z trenowaniem modelu)
```bash
python prepare_data.py  # Pobiera dane Quick Draw! (~1GB)
python train_model.py   # Trenuje model (~30 minut)
python app.py          # Uruchamia aplikację
```

## Deployment na Heroku

### 1. Przygotowanie plików
Stwórz `Procfile`:
```
web: python app.py
```

Stwórz `runtime.txt`:
```
python-3.9.18
```

### 2. Konfiguracja Heroku
```bash
# Zaloguj się do Heroku
heroku login

# Stwórz aplikację
heroku create ai-kalambury-demo

# Ustaw zmienne środowiskowe
heroku config:set FLASK_ENV=production

# Deploy
git add .
git commit -m "Initial deployment"
git push heroku main
```

### 3. Uwagi dla Heroku
- Heroku ma limit pamięci, więc użyj mniejszego modelu
- Dodaj `.slugignore` aby nie uploadować dużych plików danych
- Użyj CPU-optimized TensorFlow

## Deployment na Vercel

### 1. Struktura dla Vercel
Stwórz `vercel.json`:
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
# Zainstaluj Vercel CLI
npm i -g vercel

# Deploy
vercel
```

## Deployment na DigitalOcean App Platform

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

## Deployment na AWS EC2

### 1. Setup serwera
```bash
# Na serwerze EC2
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

## Optymalizacja dla produkcji

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
    # Cache wyników dla identycznych obrazów
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

## Bezpieczeństwo

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
    # Sprawdź rozmiar, format, etc.
    if len(image_data) > MAX_IMAGE_SIZE:
        raise ValueError("Image too large")
    # ...
```

## Troubleshooting

### Częste problemy:
1. **Out of memory** - Zmniejsz batch_size lub użyj mniejszego modelu
2. **Slow predictions** - Użyj TensorFlow Lite lub model quantization
3. **CORS errors** - Upewnij się że flask-cors jest poprawnie skonfigurowany
4. **Model not loading** - Sprawdź ścieżki do plików modelu
