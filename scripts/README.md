# Scripts

This directory contains data processing and training scripts.

## Files

- [`prepare_data.py`](prepare_data.py) - Download and preprocess Quick Draw! dataset
- [`train_model.py`](train_model.py) - Train the CNN model with comprehensive evaluation

## Usage

```bash
# 1. Prepare dataset (downloads ~2GB data)
python scripts/prepare_data.py

# 2. Train model (takes 30-60 minutes)
python scripts/train_model.py

# 3. Start API server
python app.py
```

## Configuration

Both scripts use settings from [`config.py`](../config.py) for paths and parameters.

## Output

- **prepare_data.py**: Creates `dataset/processed/` with training data
- **train_model.py**: Creates `model/drawing_model.h5` and training plots in `outputs/`
