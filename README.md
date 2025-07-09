# AI## Features

- Drawing canvas with mouse/touch support
- AI model for drawing recognition
- Real-time prediction results
- Responsive web interface
- 21 different object categories

## Technical Categories

- **Deep Learning**: Convolutional Neural Network (CNN) for image classification
- **Computer Vision**: Real-time drawing recognition and image preprocessing
- **Machine Learning**: Supervised learning with Quick Draw! dataset
- **Web Application**: Interactive Flask-based web interface
- **AI Game**: Human-AI interaction through drawing challenges

## Technologiesry

Interactive drawing recognition game where artificial intelligence tries to guess what you've drawn using deep learning and computer vision techniques.

## Features

- Drawing canvas with mouse/touch support
- AI model for drawing recognition
- Real-time prediction results
- Responsive web interface
- 21 different object categories

## Technologies

- **Frontend**: HTML, CSS, JavaScript (Canvas API for drawing)
- **Backend**: Python, Flask (RESTful API)
- **Deep Learning**: TensorFlow/Keras CNN architecture
- **Computer Vision**: PIL/Pillow for image preprocessing
- **Dataset**: Quick Draw! by Google (simplified drawings)
- **Deployment**: Production-ready with Heroku/Docker support

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Setup demo model: `python setup.py`
3. Run server: `python app.py`
4. Open browser: `http://localhost:5000`

## Full Setup (with training)

1. Download data: `python prepare_data.py`
2. Train model: `python train_model.py`
3. Run application: `python app.py`

## Project Structure

```text
ai-pictionary/
├── app.py              # Flask server
├── model/              # AI model and preprocessing
├── static/             # CSS, JS, assets
├── templates/          # HTML templates
├── data/               # Training data
└── requirements.txt    # Python dependencies
```

## Model Architecture

- **Type**: Convolutional Neural Network (CNN) for image classification
- **Input**: 28x28 grayscale images (preprocessed drawings)
- **Layers**: 3 Conv2D layers with MaxPooling for feature extraction
- **Regularization**: Dense layers with dropout for overfitting prevention
- **Output**: Softmax activation for 21-class probability distribution
- **Training**: Supervised learning on Quick Draw! simplified drawings dataset

## Supported Objects

apple, bicycle, bird, book, car, cat, chair, circle, cloud, computer, dog, flower, guitar, house, moon, phone, airplane, sun, table, tree, umbrella

## Development

### Code Quality

- Clean, readable code following Python PEP 8
- Modular architecture with separation of concerns
- Error handling and logging
- Responsive design principles

### Performance

- Optimized model inference
- Efficient image preprocessing
- Lightweight frontend assets
- Production-ready Flask configuration

## Deployment

See `DEPLOYMENT.md` for detailed deployment instructions including:

- Heroku deployment
- Docker containerization
- AWS/DigitalOcean setup
- Performance optimization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details
