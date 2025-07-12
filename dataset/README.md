# Dataset Directory

This directory contains all data files for the AI Drawing Classifier project.

## Structure

```
dataset/
├── quickdraw/          # Raw Quick Draw! .npy files
├── processed/          # Preprocessed training data
└── sample_drawings.png # Visualization of sample drawings
```

## Usage

1. **Raw data**: Downloaded automatically by `prepare_data.py`
2. **Processed data**: Created during data preparation step
3. **Sample visualization**: Generated for data inspection

## Files

- `quickdraw/*.npy` - Original Quick Draw! dataset files
- `processed/X_train.npy` - Training images
- `processed/X_val.npy` - Validation images
- `processed/X_test.npy` - Test images
- `processed/y_train.npy` - Training labels
- `processed/y_val.npy` - Validation labels
- `processed/y_test.npy` - Test labels
- `processed/classes.json` - Class mapping
