# Model Training Guide

This guide provides comprehensive instructions for training and improving the AI Drawing Classifier model with **balanced performance across all classes**.

## 🎯 Training Philosophy

Our training approach focuses on:
- **Equal class performance** - No single class is prioritized over others
- **Balanced optimization** - Uses class weights to handle any dataset imbalances
- **Comprehensive evaluation** - Per-class metrics to identify strengths and weaknesses
- **Iterative improvement** - Clear guidance on enhancing model performance

## 📊 Overview

The model is a Convolutional Neural Network (CNN) trained on Google's Quick Draw! dataset. It achieves balanced performance across 50 different drawing categories with optimized class weighting.

## 🗂️ Dataset

### Quick Draw! Dataset
- **Source**: Google's Quick Draw!
- **Format**: 28x28 grayscale bitmap images
- **Classes**: 50 predefined categories
- **Size**: ~5,000 samples per class (configurable)

### Supported Classes
```
apple, bicycle, bird, book, car, cat, chair, circle, cloud, computer,
dog, flower, guitar, house, moon, phone, airplane, sun, table, tree,
umbrella, fish, bus, clock, coffee, cup, elephant, eye, face, fork,
hand, hat, heart, key, knife, lamp, leaf, mountain, mouse, mushroom,
pencil, pizza, rainbow, shoe, snake, star, sword, train, truck, whale
```

## 🚀 Quick Training Setup

### 1. Download Dataset
```bash
# Download Quick Draw! data for all classes
python prepare_data.py
```

This will:
- Download `.npy` files for each class
- Store data in `data/quickdraw/` directory
- Each file contains thousands of 28x28 drawings

### 2. Train Model
```bash
# Train with default parameters
python train_model.py
```

Training output:
- Model saved to `model/drawing_model.h5`
- Classes saved to `model/classes.json`
- Training plots in `outputs/training_history.png`

## ⚖️ Balanced Training Features

### Class Balancing
The training script automatically:
- **Analyzes class distribution** in your dataset
- **Applies class weights** to balance training for underrepresented classes
- **Reports balance metrics** showing performance equality across classes
- **Identifies classes needing improvement** for targeted optimization

### Performance Analysis
During training, you'll see:
```
📊 Class Distribution Analysis:
⚖️  Class balance ratio: 0.987 (1.0 = perfect balance)
✅ Dataset is reasonably balanced

🎯 Balanced Model Performance Analysis
📈 Balance Metrics:
Mean class accuracy: 0.8234 ± 0.0456
✅ Excellent balance - consistent performance across classes

🏆 Top 5 performing classes:
  1. circle: 0.892
  2. table: 0.875
  3. house: 0.864

🎯 Classes needing improvement:
  mouse: 0.743
  snake: 0.756
```

### 3. Test Model
```bash
# Start API to test predictions
python app.py
```

## 🔧 Advanced Training Configuration

### Customizing Data Preparation

Edit `prepare_data.py` to modify data collection:

```python
# Modify these parameters in prepare_data.py

# Number of samples per class (default: 5000)
max_samples_per_class = 10000  # More data = better accuracy

# Add new classes
CLASSES = [
    'apple', 'bicycle', 'bird', 'book', 'car', 'cat', 'chair', 'circle',
    'cloud', 'computer', 'dog', 'flower', 'guitar', 'house', 'moon',
    'phone', 'airplane', 'sun', 'table', 'tree', 'umbrella',
    'clock', 'fish', 'star'  # Add new classes here
]

# Data augmentation settings
apply_augmentation = True
rotation_range = 15
zoom_range = 0.1
width_shift_range = 0.1
height_shift_range = 0.1
```

### Model Architecture Customization

Edit `train_model.py` to modify the CNN architecture:

```python
def create_model(num_classes):
    model = models.Sequential([
        # Input layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        # Add more convolutional layers for better feature extraction
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Optional: Add batch normalization
        layers.BatchNormalization(),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Prevent overfitting

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model
```

### Hyperparameter Tuning

Key parameters to adjust in `train_model.py`:

```python
# Training parameters
BATCH_SIZE = 128        # Larger = faster training, more memory
EPOCHS = 50            # More epochs = potentially better accuracy
LEARNING_RATE = 0.001  # Lower = more stable, slower convergence
VALIDATION_SPLIT = 0.2 # 20% data for validation

# Optimizer settings
optimizer = 'adam'     # Options: 'adam', 'sgd', 'rmsprop'

# Loss function
loss = 'sparse_categorical_crossentropy'  # For integer labels

# Callbacks
callbacks = [
    # Stop training when validation loss stops improving
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),

    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    ),

    # Save best model during training
    ModelCheckpoint(
        'model/drawing_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]
```

## 📈 Performance Optimization

### 1. Data Augmentation

Increase training data variety:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,          # Rotate images
    width_shift_range=0.2,      # Horizontal shift
    height_shift_range=0.2,     # Vertical shift
    zoom_range=0.2,             # Zoom in/out
    horizontal_flip=False,      # Don't flip (drawings have orientation)
    fill_mode='constant',       # Fill empty pixels with black
    cval=0
)
```

### 2. Transfer Learning

Use pre-trained features:

```python
# Load pre-trained model as base
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(28, 28, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Freeze base layers

model = tf.keras.Sequential([
    # Convert grayscale to RGB
    layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1)),

    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])
```

### 3. Advanced Architectures

Try different model architectures:

```python
# ResNet-style skip connections
def create_resnet_model(num_classes):
    inputs = layers.Input(shape=(28, 28, 1))

    # Initial conv layer
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)

    # Residual block
    shortcut = x
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    # Continue with more layers...
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)
```

## 🔍 Evaluation and Analysis

### Training Metrics

Monitor these metrics during training:

```python
# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('model/training_plots/training_history.png')
    plt.show()
```

### Model Analysis

Evaluate model performance:

```python
# Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Generate predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
print(classification_report(y_test, y_pred_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('model/training_plots/confusion_matrix.png')
plt.show()
```

### Error Analysis

Identify problematic cases:

```python
# Find misclassified examples
def analyze_errors(model, X_test, y_test, class_names):
    predictions = model.predict(X_test)
    pred_classes = np.argmax(predictions, axis=1)

    # Find misclassified samples
    errors = pred_classes != y_test
    error_indices = np.where(errors)[0]

    # Show worst predictions
    confidences = np.max(predictions, axis=1)
    worst_indices = error_indices[np.argsort(confidences[errors])[::-1]]

    # Plot examples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, idx in enumerate(worst_indices[:10]):
        ax = axes[i//5, i%5]
        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'True: {class_names[y_test[idx]]}\n'
                    f'Pred: {class_names[pred_classes[idx]]}\n'
                    f'Conf: {confidences[idx]:.2f}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('model/training_plots/error_analysis.png')
    plt.show()
```

## 🆕 Adding New Classes

### Step 1: Download New Data

Add new class to `CLASSES` list in `prepare_data.py`:

```python
CLASSES = [
    # Existing classes...
    'umbrella',
    # New classes
    'clock',
    'fish',
    'star',
    'pizza'
]
```

### Step 2: Verify Data Availability

Check if class exists in Quick Draw! dataset:
```bash
# Test URL manually
curl -I https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/clock.npy
```

### Step 3: Retrain Model

```bash
# Download new data
python prepare_data.py

# Retrain with new classes
python train_model.py
```

### Step 4: Update Classifier

Update `model/drawing_classifier.py` with new class list:

```python
class DrawingClassifier:
    def __init__(self):
        self.classes = [
            'apple', 'bicycle', 'bird', 'book', 'car', 'cat', 'chair',
            'circle', 'cloud', 'computer', 'dog', 'flower', 'guitar',
            'house', 'moon', 'phone', 'airplane', 'sun', 'table',
            'tree', 'umbrella',
            # New classes
            'clock', 'fish', 'star', 'pizza'
        ]
```

## 🚀 Production Optimization

### Model Size Reduction

Reduce model size for deployment:

```python
# Quantization (reduce model size)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save quantized model
with open('model/drawing_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Model Pruning

Remove unnecessary weights:

```python
import tensorflow_model_optimization as tfmot

# Apply pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Prune model
pruned_model = prune_low_magnitude(
    model,
    pruning_schedule=tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.50,
        final_sparsity=0.80,
        begin_step=0,
        end_step=1000
    )
)

# Compile and train pruned model
pruned_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
```

## 📊 Benchmarking

Compare model versions:

```python
# Benchmark script
import time

def benchmark_model(model, test_data, num_runs=100):
    """Benchmark model inference speed"""

    # Warmup
    for _ in range(10):
        model.predict(test_data[:1])

    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        predictions = model.predict(test_data[:1])
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    print(f"Average inference time: {avg_time*1000:.2f}ms")

    return avg_time
```

## 🔄 Continuous Improvement

### A/B Testing

Test model versions:

```python
# Model versioning
def save_model_version(model, version):
    model.save(f'model/drawing_model_v{version}.h5')

# Load specific version
def load_model_version(version):
    return tf.keras.models.load_model(f'model/drawing_model_v{version}.h5')
```

### Performance Monitoring

Track model performance over time:

```python
# Log predictions for analysis
import json
from datetime import datetime

def log_prediction(image_data, prediction, confidence, true_label=None):
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'prediction': prediction,
        'confidence': float(confidence),
        'true_label': true_label,
        'model_version': '1.0'
    }

    with open('logs/predictions.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
```

## 🛠️ Troubleshooting

### Common Issues

1. **Low accuracy**: Try more data, data augmentation, or deeper model
2. **Overfitting**: Add dropout, reduce model complexity, or more data
3. **Slow training**: Reduce batch size, use GPU, or optimize data pipeline
4. **Memory issues**: Reduce batch size or use data generators

### Debug Training

```python
# Check data distribution
def analyze_data_distribution(y_train, class_names):
    unique, counts = np.unique(y_train, return_counts=True)
    for i, count in enumerate(counts):
        print(f"{class_names[i]}: {count} samples")

    # Plot distribution
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, counts)
    plt.xticks(rotation=45)
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.show()
```

## 📚 Resources

- [Quick Draw! Dataset](https://quickdraw.withgoogle.com/data)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [CNN Architecture Guide](https://cs231n.github.io/convolutional-networks/)
- [Model Optimization Techniques](https://www.tensorflow.org/model_optimization)

## 🤝 Contributing

To contribute model improvements:

1. Create experiments in `experiments/` directory
2. Document changes and results
3. Submit pull request with performance comparison
4. Include training logs and validation metrics

## 📞 Support

For training issues or questions:
- Open GitHub issue with training logs
- Include system specifications and error messages
- Provide sample data if possible
