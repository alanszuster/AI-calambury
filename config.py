"""
Configuration file for AI Drawing Classifier project.
"""

import os

# Project structure
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, 'experiments')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
DOCKER_DIR = os.path.join(PROJECT_ROOT, 'docker')

# Dataset configuration
QUICKDRAW_DIR = os.path.join(DATASET_DIR, 'quickdraw')
PROCESSED_DIR = os.path.join(DATASET_DIR, 'processed')
QUICKDRAW_BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

# Model configuration
MODEL_PATH = os.path.join(MODEL_DIR, 'drawing_model.h5')
CLASSES_PATH = os.path.join(MODEL_DIR, 'classes.json')

# Training configuration
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MAX_SAMPLES_PER_CLASS = 5000

# Data split configuration
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_SEED = 42

# Classes to train on (verified as available in Quick Draw! dataset)
CLASSES = [
    'apple', 'bicycle', 'bird', 'book', 'car', 'cat', 'chair', 'circle',
    'cloud', 'computer', 'dog', 'flower', 'guitar', 'house', 'moon',
    'airplane', 'sun', 'table', 'tree', 'umbrella', 'fish',
    'bus', 'clock', 'cup', 'elephant', 'eye', 'face', 'fork',
    'hand', 'hat', 'key', 'knife', 'leaf', 'mountain',
    'mouse', 'mushroom', 'pencil', 'pizza', 'rainbow', 'shoe', 'snake',
    'star', 'sword', 'train', 'truck', 'whale'
]

# Note: These classes are not available in Quick Draw! dataset:
# 'phone', 'coffee', 'heart', 'lamp'
