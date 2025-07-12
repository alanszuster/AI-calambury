import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import json

def create_model(num_classes):
    """Create an improved CNN model with better architecture for balanced training"""
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Third convolutional block - reduced size
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def load_prepared_data():
    data_dir = 'dataset/processed'

    print("Loading data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    with open(os.path.join(data_dir, 'classes.json'), 'r') as f:
        class_mapping = json.load(f)    # Get all unique labels across datasets
    all_labels = np.concatenate([y_train, y_val, y_test])
    unique_labels = sorted(np.unique(all_labels))

    print(f"Original labels range: {min(unique_labels)} to {max(unique_labels)}")
    print(f"Found {len(unique_labels)} unique labels in data")

    # Filter out labels that don't have class names
    valid_labels = [label for label in unique_labels if str(label) in class_mapping]
    invalid_labels = [label for label in unique_labels if str(label) not in class_mapping]

    if invalid_labels:
        print(f"⚠️  Warning: Found labels without class names: {invalid_labels}")
        print(f"⚠️  These will be filtered out from training data")

    print(f"Using {len(valid_labels)} valid labels: {valid_labels[:10]}...")

    # Filter data to only include samples with valid labels
    def filter_data(X, y, valid_labels):
        mask = np.isin(y, valid_labels)
        return X[mask], y[mask]

    X_train_filtered, y_train_filtered = filter_data(X_train, y_train, valid_labels)
    X_val_filtered, y_val_filtered = filter_data(X_val, y_val, valid_labels)
    X_test_filtered, y_test_filtered = filter_data(X_test, y_test, valid_labels)

    # Create mapping from original labels to continuous range [0, n-1]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_labels)}

    # Remap labels to continuous range
    def remap_labels(y, mapping):
        return np.array([mapping[label] for label in y])

    y_train_remapped = remap_labels(y_train_filtered, label_mapping)
    y_val_remapped = remap_labels(y_val_filtered, label_mapping)
    y_test_remapped = remap_labels(y_test_filtered, label_mapping)

    # Create new class mapping for remapped labels
    new_class_mapping = {}
    for new_label, old_label in enumerate(valid_labels):
        new_class_mapping[str(new_label)] = class_mapping[str(old_label)]

    print(f"Remapped {len(valid_labels)} classes to continuous range: 0 to {len(valid_labels)-1}")
    print(f"Data loaded:")
    print(f"  Training: {X_train_filtered.shape} (filtered from {X_train.shape})")
    print(f"  Validation: {X_val_filtered.shape} (filtered from {X_val.shape})")
    print(f"  Test: {X_test_filtered.shape} (filtered from {X_test.shape})")
    print(f"  Classes: {len(new_class_mapping)}")

    return X_train_filtered, X_val_filtered, X_test_filtered, y_train_remapped, y_val_remapped, y_test_remapped, new_class_mapping

def analyze_class_distribution(y_train, y_val, classes):
    """Analyze the distribution of classes in training and validation sets"""
    print("\n📊 Class Distribution Analysis:")
    print("=" * 60)

    train_counts = np.bincount(y_train)
    val_counts = np.bincount(y_val)

    print(f"{'Class':<15} {'Train':<8} {'Val':<8} {'Total':<8}")
    print("-" * 45)

    for i, class_name in enumerate(classes):
        train_count = train_counts[i] if i < len(train_counts) else 0
        val_count = val_counts[i] if i < len(val_counts) else 0
        total = train_count + val_count
        print(f"{class_name:<15} {train_count:<8} {val_count:<8} {total:<8}")

    print("-" * 45)
    print(f"{'TOTAL':<15} {len(y_train):<8} {len(y_val):<8} {len(y_train) + len(y_val):<8}")

    # Check balance
    if len(train_counts) > 0:
        min_count = np.min(train_counts)
        max_count = np.max(train_counts)
        balance_ratio = min_count / max_count if max_count > 0 else 0
        print(f"\n⚖️  Class balance ratio: {balance_ratio:.3f} (1.0 = perfect balance)")

        if balance_ratio < 0.8:
            print("⚠️  Imbalanced dataset detected - using class weights for training")
        else:
            print("✅ Dataset is reasonably balanced")

def train_model(X_train, X_val, y_train, y_val, num_classes, epochs=50):
    print("\nCreating balanced model for all classes...")
    model = create_model(num_classes)

    # Use class weights to balance training
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    print(f"Applied class balancing for {num_classes} classes")

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Enhanced callbacks for balanced training
    checkpoint = ModelCheckpoint(
        'model/best_model.keras',  # Using native Keras format
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Increased patience for better convergence
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,   # Increased patience
        min_lr=0.00001,
        verbose=1
    )

    print(f"\nStarting balanced training for {epochs} epochs...")
    print("🎯 Goal: High performance across ALL classes")

    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,  # Apply class balancing
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )

    return model, history

def evaluate_balanced_performance(model, X_test, y_test, classes):
    """Comprehensive evaluation focusing on balanced performance across all classes"""
    print("\n🎯 Balanced Model Performance Analysis")
    print("=" * 60)

    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    # Overall accuracy
    overall_accuracy = np.mean(y_pred == y_test)
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

    # Per-class analysis
    class_accuracies = []
    class_counts = []

    print(f"\n{'Class':<15} {'Accuracy':<10} {'Count':<8} {'Performance'}")
    print("-" * 50)

    for i, class_name in enumerate(classes):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
            class_count = np.sum(class_mask)
            class_accuracies.append(class_acc)
            class_counts.append(class_count)

            # Performance indicator
            if class_acc >= 0.8:
                perf = "🟢 Excellent"
            elif class_acc >= 0.6:
                perf = "🟡 Good"
            elif class_acc >= 0.4:
                perf = "🟠 Fair"
            else:
                perf = "🔴 Needs work"

            print(f"{class_name:<15} {class_acc:<10.3f} {class_count:<8} {perf}")

    # Summary statistics
    if class_accuracies:
        mean_acc = np.mean(class_accuracies)
        std_acc = np.std(class_accuracies)
        min_acc = np.min(class_accuracies)
        max_acc = np.max(class_accuracies)

        print("\n📈 Balance Metrics:")
        print(f"Mean class accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Best performing class: {max_acc:.4f}")
        print(f"Worst performing class: {min_acc:.4f}")
        print(f"Performance range: {max_acc - min_acc:.4f}")

        # Balance assessment
        if std_acc < 0.1:
            print("✅ Excellent balance - consistent performance across classes")
        elif std_acc < 0.15:
            print("🟡 Good balance - minor performance variations")
        else:
            print("🟠 Moderate imbalance - some classes need improvement")

        # Show top and bottom performers
        class_performance = list(zip(classes, class_accuracies))
        class_performance.sort(key=lambda x: x[1], reverse=True)

        print(f"\n🏆 Top 5 performing classes:")
        for i, (class_name, acc) in enumerate(class_performance[:5]):
            print(f"  {i+1}. {class_name}: {acc:.3f}")

        print(f"\n🎯 Classes needing improvement:")
        for i, (class_name, acc) in enumerate(class_performance[-5:]):
            print(f"  {class_name}: {acc:.3f}")

    return {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': class_accuracies,
        'mean_class_accuracy': mean_acc if class_accuracies else 0,
        'std_class_accuracy': std_acc if class_accuracies else 0
    }

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
    plt.savefig('outputs/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Training plots saved to outputs/training_history.png")

def save_final_model(model, class_mapping):
    model.save('model/drawing_model.keras')  # Using native Keras format

    with open('model/classes.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)

    print("✅ Model saved to model/drawing_model.keras")
    print("✅ Class mapping saved to model/classes.json")

def validate_environment():
    """Validate that required packages are available."""
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")

        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("⚠️  No GPU detected, training will use CPU (slower)")

        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        return False

def main():
    print("🎨 AI Pictionary - Balanced Model Training")
    print("=" * 50)
    print("🎯 Goal: High-quality performance across ALL classes")

    if not os.path.exists('dataset/processed'):
        print("❌ No processed data found!")
        print("Please run: python scripts/prepare_data.py first")
        return

    os.makedirs('model', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    try:
        # Validate environment before proceeding
        if not validate_environment():
            print("❌ Environment validation failed. Please fix the issues above.")
            return

        # Load data with automatic label remapping
        X_train, X_val, X_test, y_train, y_val, y_test, class_mapping = load_prepared_data()

        # Number of classes is simply the length of the class mapping
        num_classes = len(class_mapping)
        classes = list(class_mapping.values())

        print(f"Training model with {num_classes} classes: {classes[:5]}...")

        # Analyze class distribution
        analyze_class_distribution(y_train, y_val, classes)

        # Train balanced model
        model, history = train_model(X_train, X_val, y_train, y_val, num_classes)

        # Evaluate with balanced metrics
        evaluate_balanced_performance(model, X_test, y_test, classes)

        # Plot training history
        try:
            plot_training_history(history)
        except Exception as e:
            print(f"Cannot create plots: {e}")

        # Save final model
        save_final_model(model, class_mapping)

        print("\n🎉 Balanced training completed successfully!")
        print("✅ Model optimized for equal performance across all classes")
        print("You can now run the application: python app.py")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
