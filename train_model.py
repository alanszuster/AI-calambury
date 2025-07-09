import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json

def create_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def load_prepared_data():
    data_dir = 'data/processed'

    print("Loading data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    with open(os.path.join(data_dir, 'classes.json'), 'r') as f:
        class_mapping = json.load(f)

    print(f"Data loaded:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Classes: {len(class_mapping)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, class_mapping

def train_model(X_train, X_val, y_train, y_val, num_classes, epochs=50):
    print("\nCreating model...")
    model = create_model(num_classes)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    checkpoint = ModelCheckpoint(
        'model/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )

    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )

    return model, history

def evaluate_model(model, X_test, y_test, class_mapping):
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    predictions = model.predict(X_test[:10])
    predicted_classes = np.argmax(predictions, axis=1)

    print("\nSample predictions:")
    for i in range(10):
        true_class = class_mapping[str(y_test[i])]
        pred_class = class_mapping[str(predicted_classes[i])]
        confidence = predictions[i][predicted_classes[i]] * 100

        status = "✅" if true_class == pred_class else "❌"
        print(f"  {status} True: {true_class:12} | Predicted: {pred_class:12} | Confidence: {confidence:.1f}%")

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Training plots saved to model/training_history.png")

def save_final_model(model, class_mapping):
    model.save('model/drawing_model.h5')

    with open('model/classes.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)

    print("✅ Model saved to model/drawing_model.h5")
    print("✅ Class mapping saved to model/classes.json")

def main():
    print("🎨 AI Pictionary - Model Training")
    print("=" * 50)

    if not os.path.exists('data/processed'):
        print("❌ No processed data found!")
        print("Please run: python prepare_data.py first")
        return

    os.makedirs('model', exist_ok=True)

    try:
        X_train, X_val, X_test, y_train, y_val, y_test, class_mapping = load_prepared_data()
        num_classes = len(class_mapping)

        model, history = train_model(X_train, X_val, y_train, y_val, num_classes)

        evaluate_model(model, X_test, y_test, class_mapping)

        try:
            plot_training_history(history)
        except Exception as e:
            print(f"Cannot create plots: {e}")

        save_final_model(model, class_mapping)

        print("\n🎉 Training completed successfully!")
        print("You can now run the application: python app.py")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
