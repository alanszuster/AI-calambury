import os
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split

CLASSES = [
    'apple', 'bicycle', 'bird', 'book', 'car', 'cat', 'chair', 'circle',
    'cloud', 'computer', 'dog', 'flower', 'guitar', 'house', 'moon',
    'phone', 'airplane', 'sun', 'table', 'tree', 'umbrella', 'fish',
    'bus', 'clock', 'coffee', 'cup', 'elephant', 'eye', 'face', 'fork',
    'hand', 'hat', 'heart', 'key', 'knife', 'lamp', 'leaf', 'mountain',
    'mouse', 'mushroom', 'pencil', 'pizza', 'rainbow', 'shoe', 'snake',
    'star', 'sword', 'train', 'truck', 'whale'
]

BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

def download_data(classes, data_dir='dataset/quickdraw'):
    """Download Quick Draw! dataset files for specified classes.

    Args:
        classes (list): List of class names to download
        data_dir (str): Directory to store downloaded files
    """
    os.makedirs(data_dir, exist_ok=True)

    failed_downloads = []

    for class_name in classes:
        print(f"Downloading {class_name}...")

        url = f"{BASE_URL}{class_name}.npy"
        file_path = os.path.join(data_dir, f"{class_name}.npy")

        if os.path.exists(file_path):
            print(f"  ✓ {class_name}.npy already exists")
            continue

        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"  ✓ Downloaded {class_name}.npy")

        except Exception as e:
            print(f"  ❌ Error downloading {class_name}: {e}")
            failed_downloads.append(class_name)

    if failed_downloads:
        print(f"\n⚠️  Failed to download: {failed_downloads}")
        print("Please check your internet connection and try again.")

    return failed_downloads

def load_quickdraw_data(classes, data_dir='dataset/quickdraw', max_samples_per_class=5000):
    """Load and preprocess Quick Draw! dataset.

    Args:
        classes (list): List of class names to load
        data_dir (str): Directory containing downloaded .npy files
        max_samples_per_class (int): Maximum samples per class

    Returns:
        tuple: (X, y, classes) where X is image data, y is labels
    """
    X = []
    y = []
    missing_files = []

    for class_idx, class_name in enumerate(classes):
        file_path = os.path.join(data_dir, f"{class_name}.npy")

        if not os.path.exists(file_path):
            print(f"❌ Missing file for {class_name}")
            missing_files.append(class_name)
            continue

        print(f"Loading data for {class_name}...")

        try:
            data = np.load(file_path)

            # Ensure we don't exceed available data
            actual_samples = min(len(data), max_samples_per_class)

            if len(data) > max_samples_per_class:
                # Use random seed for reproducibility
                np.random.seed(42)
                indices = np.random.choice(len(data), max_samples_per_class, replace=False)
                data = data[indices]

            X.append(data)
            y.extend([class_idx] * len(data))

            print(f"  ✓ Loaded {len(data)} samples")

        except Exception as e:
            print(f"  ❌ Error loading {class_name}: {e}")
            missing_files.append(class_name)

    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")

    if not X:
        raise ValueError("No data was loaded! Please check your data directory.")

    X = np.concatenate(X, axis=0)
    y = np.array(y)

    # Reshape and normalize
    X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    print(f"\n📊 Data summary:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of classes: {len(classes) - len(missing_files)}")
    print(f"Total samples: {len(X)}")

    return X, y, [c for c in classes if c not in missing_files]

def visualize_samples(X, y, classes, n_samples=5):
    fig, axes = plt.subplots(len(classes), n_samples, figsize=(15, len(classes) * 2))

    for class_idx, class_name in enumerate(classes):
        class_indices = np.where(y == class_idx)[0]
        sample_indices = np.random.choice(class_indices, min(n_samples, len(class_indices)), replace=False)

        for i, sample_idx in enumerate(sample_indices):
            ax = axes[class_idx, i] if len(classes) > 1 else axes[i]
            ax.imshow(X[sample_idx].reshape(28, 28), cmap='gray')
            ax.set_title(f"{class_name}" if i == 0 else "")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('dataset/sample_drawings.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Sample drawings saved to dataset/sample_drawings.png")

def prepare_data_for_training(X, y, test_size=0.2, val_size=0.1):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )

    print(f"\n📊 Data split:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """Main function to prepare data for training."""
    print("🎨 AI Drawing Classifier - Data Preparation")
    print("=" * 50)

    print("\n1. Downloading Quick Draw! data...")
    failed_downloads = download_data(CLASSES)

    if failed_downloads:
        print(f"⚠️  Some downloads failed. Continuing with available data...")

    print("\n2. Loading and processing data...")
    try:
        X, y, successful_classes = load_quickdraw_data(CLASSES)
    except ValueError as e:
        print(f"❌ {e}")
        return False

    print("\n3. Visualizing sample drawings...")
    try:
        visualize_samples(X, y, successful_classes)
    except Exception as e:
        print(f"Cannot create visualization: {e}")

    print("\n4. Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_for_training(X, y)

    print("\n5. Saving processed data...")
    data_dir = 'dataset/processed'
    os.makedirs(data_dir, exist_ok=True)

    try:
        np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(data_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(data_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

        # Use successful classes for mapping
        class_mapping = {i: class_name for i, class_name in enumerate(successful_classes)}
        with open(os.path.join(data_dir, 'classes.json'), 'w') as f:
            json.dump(class_mapping, f, indent=2)

        print(f"✅ Data prepared and saved to {data_dir}")
        print(f"✅ Successfully processed {len(successful_classes)} classes")
        print("\nYou can now run train_model.py to train the model!")
        return True

    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return False

if __name__ == "__main__":
    main()
