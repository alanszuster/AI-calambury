import tensorflow as tf
import numpy as np
from PIL import Image
import os

class DrawingClassifier:
    def __init__(self):

        self.model = None
        self.classes = [
            'apple', 'bicycle', 'bird', 'book', 'car', 'cat', 'chair', 'circle',
            'cloud', 'computer', 'dog', 'flower', 'guitar', 'house', 'moon',
            'phone', 'plane', 'sun', 'table', 'tree', 'umbrella'
        ]
        self.english_classes = [
            'apple', 'bicycle', 'bird', 'book', 'car', 'cat', 'chair', 'circle',
            'cloud', 'computer', 'dog', 'flower', 'guitar', 'house', 'moon',
            'phone', 'airplane', 'sun', 'table', 'tree', 'umbrella'
        ]

    def create_simple_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(self.classes), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        image = image.convert('L')
        img_array = np.array(image)
        img_array = 255 - img_array
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        return img_array

    def load_model(self):
        model_path = 'model/drawing_model.h5'

        if os.path.exists(model_path):
            print("Loading existing model...")
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("Creating new model...")
            self.model = self.create_simple_model()
            print("Model created with random weights. Requires training for accuracy.")

    def predict(self, image):
        if self.model is None:
            return [{'class': 'error', 'confidence': 0.0}]

        try:
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
            top_indices = np.argsort(predictions[0])[::-1][:3]

            results = []
            for i, idx in enumerate(top_indices):
                confidence = float(predictions[0][idx])
                class_name = self.english_classes[idx] if idx < len(self.english_classes) else self.classes[idx]
                results.append({
                    'class': class_name,
                    'confidence': round(confidence * 100, 1)
                })

            return results

        except Exception as e:
            print(f"Error during prediction: {e}")
            return [{'class': 'error', 'confidence': 0.0}]

    def save_model(self, path='model/drawing_model.h5'):
        if self.model is not None:
            self.model.save(path)
            print(f"Model saved to {path}")
