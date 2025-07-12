# API Documentation

## OpenAPI Specification

```yaml
openapi: 3.0.0
info:
  title: AI Drawing Classifier API
  description: Backend API for real-time drawing recognition using deep learning
  version: 1.0.0
  contact:
    name: API Support
    url: https://github.com/your-repo/issues

servers:
  - url: http://localhost:5000
    description: Development server
  - url: https://your-app.herokuapp.com
    description: Production server

paths:
  /:
    get:
      summary: Get API information
      description: Returns API metadata and available endpoints
      responses:
        '200':
          description: API information
          content:
            application/json:
              schema:
                type: object
                properties:
                  name:
                    type: string
                    example: "AI Drawing Classifier API"
                  version:
                    type: string
                    example: "1.0"
                  endpoints:
                    type: object
                    properties:
                      "POST /predict":
                        type: string
                        example: "Classify a drawing image"
                      "GET /classes":
                        type: string
                        example: "Get list of supported classes"
                      "GET /health":
                        type: string
                        example: "Check API health"

  /predict:
    post:
      summary: Classify drawing image
      description: Upload a drawing image and get AI predictions
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - image
              properties:
                image:
                  type: string
                  description: Base64 encoded image data with data URL prefix
                  example: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
      responses:
        '200':
          description: Prediction results
          content:
            application/json:
              schema:
                type: object
                properties:
                  predictions:
                    type: array
                    items:
                      type: object
                      properties:
                        class:
                          type: string
                          example: "cat"
                        confidence:
                          type: number
                          format: float
                          minimum: 0
                          maximum: 1
                          example: 0.92
                  success:
                    type: boolean
                    example: true
        '400':
          description: Bad request (invalid image data)
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '500':
          description: Internal server error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /classes:
    get:
      summary: Get supported classes
      description: Returns list of all drawing classes the model can recognize
      responses:
        '200':
          description: List of supported classes
          content:
            application/json:
              schema:
                type: object
                properties:
                  classes:
                    type: array
                    items:
                      type: string
                    example: ["apple", "bicycle", "bird", "book", "car"]
                  total_classes:
                    type: integer
                    example: 21
        '500':
          description: Model not loaded
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /health:
    get:
      summary: Health check
      description: Check API health and model status
      responses:
        '200':
          description: Health status
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    enum: [healthy, unhealthy]
                    example: "healthy"
                  model:
                    type: string
                    enum: [loaded, not_loaded]
                    example: "loaded"
                  version:
                    type: string
                    example: "1.0"

  /get_random_word:
    get:
      summary: Get random class
      description: Returns a random drawing class for challenges
      responses:
        '200':
          description: Random class
          content:
            application/json:
              schema:
                type: object
                properties:
                  word:
                    type: string
                    example: "cat"
        '500':
          description: Model not loaded
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

components:
  schemas:
    Error:
      type: object
      properties:
        error:
          type: string
          description: Error message
          example: "Model not loaded"
        predictions:
          type: array
          items: {}
          description: Empty array for failed predictions
          example: []
```

## Usage Examples

### Python

```python
import requests
import base64
from PIL import Image
import io

# Read and encode image
def encode_image(image_path):
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{image_data}'

# Make prediction
def predict_drawing(image_path):
    url = 'http://localhost:5000/predict'

    payload = {
        'image': encode_image(image_path)
    }

    response = requests.post(url, json=payload)
    result = response.json()

    if result.get('success'):
        top_prediction = result['predictions'][0]
        print(f"Prediction: {top_prediction['class']}")
        print(f"Confidence: {top_prediction['confidence']:.2%}")
    else:
        print(f"Error: {result.get('error')}")

# Example usage
predict_drawing('my_drawing.png')
```

### JavaScript (Node.js)

```javascript
const axios = require('axios');
const fs = require('fs');

async function predictDrawing(imagePath) {
    try {
        // Read and encode image
        const imageBuffer = fs.readFileSync(imagePath);
        const imageBase64 = imageBuffer.toString('base64');
        const imageData = `data:image/png;base64,${imageBase64}`;

        // Make request
        const response = await axios.post('http://localhost:5000/predict', {
            image: imageData
        });

        const result = response.data;

        if (result.success) {
            const topPrediction = result.predictions[0];
            console.log(`Prediction: ${topPrediction.class}`);
            console.log(`Confidence: ${(topPrediction.confidence * 100).toFixed(1)}%`);
        } else {
            console.log(`Error: ${result.error}`);
        }

    } catch (error) {
        console.error('Request failed:', error.message);
    }
}

// Example usage
predictDrawing('my_drawing.png');
```

### JavaScript (Browser)

```javascript
// Assuming you have a canvas element with a drawing
function predictCanvasDrawing(canvas) {
    // Convert canvas to base64
    const imageData = canvas.toDataURL('image/png');

    // Make request
    fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: imageData
        })
    })
    .then(response => response.json())
    .then(result => {
        if (result.success) {
            const topPrediction = result.predictions[0];
            console.log(`Prediction: ${topPrediction.class}`);
            console.log(`Confidence: ${(topPrediction.confidence * 100).toFixed(1)}%`);
        } else {
            console.error(`Error: ${result.error}`);
        }
    })
    .catch(error => {
        console.error('Request failed:', error);
    });
}
```

### cURL

```bash
# Get API info
curl -X GET http://localhost:5000/

# Get supported classes
curl -X GET http://localhost:5000/classes

# Health check
curl -X GET http://localhost:5000/health

# Get random word
curl -X GET http://localhost:5000/get_random_word

# Predict drawing (with base64 encoded image)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."}'
```

## Response Examples

### Successful Prediction

```json
{
  "predictions": [
    {
      "class": "cat",
      "confidence": 0.8234567
    },
    {
      "class": "dog",
      "confidence": 0.1123456
    },
    {
      "class": "bird",
      "confidence": 0.0456789
    }
  ],
  "success": true
}
```

### Error Response

```json
{
  "error": "Model not loaded",
  "predictions": []
}
```

### Classes Response

```json
{
  "classes": [
    "apple", "bicycle", "bird", "book", "car", "cat", "chair",
    "circle", "cloud", "computer", "dog", "flower", "guitar",
    "house", "moon", "phone", "airplane", "sun", "table",
    "tree", "umbrella"
  ],
  "total_classes": 21
}
```

## Error Handling

### Common Errors

1. **Model not loaded**: API started but model failed to initialize
2. **Invalid image data**: Malformed base64 or unsupported image format
3. **Missing image field**: Request body doesn't contain required 'image' field
4. **Large image**: Image too large (recommended max: 5MB)

### Error Response Format

All errors return HTTP status codes with JSON response:

```json
{
  "error": "Descriptive error message",
  "predictions": []
}
```

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error (model issues)

## Rate Limiting

Currently no rate limiting is implemented. For production usage, consider:

- Adding rate limiting middleware
- Implementing API keys for access control
- Setting up request quotas per client

## CORS Configuration

The API includes CORS headers to allow cross-origin requests from web browsers. Current configuration allows all origins (`*`).

For production, configure specific allowed origins:

```python
from flask_cors import CORS

# Allow specific origins only
CORS(app, origins=['https://your-frontend-domain.com'])
```

## Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 28x28 grayscale images
- **Output**: 21 class probabilities
- **Model Size**: ~2MB
- **Inference Time**: <100ms average

## Deployment Notes

### Environment Variables

The API can be configured using environment variables:

```bash
export FLASK_ENV=production
export PORT=5000
export MODEL_PATH=model/drawing_model.h5
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
```

### Health Monitoring

Use the `/health` endpoint for:
- Load balancer health checks
- Monitoring system integration
- Automated deployment verification
