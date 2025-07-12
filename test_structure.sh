#!/bin/bash

# Quick test script to verify project structure and imports

echo "🧪 Testing AI Drawing Classifier Project Structure"
echo "================================================="

echo ""
echo "📁 Checking project structure..."
if [ -d "scripts" ] && [ -d "docs" ] && [ -d "docker" ] && [ -d "model" ] && [ -d "dataset" ]; then
    echo "✅ All required directories exist"
else
    echo "❌ Missing required directories"
    exit 1
fi

echo ""
echo "🐍 Testing Python imports..."

# Test config import
python -c "import config; print('✅ config.py imports successfully')" || echo "❌ config.py import failed"

# Test app import
python -c "from model.drawing_classifier import DrawingClassifier; print('✅ DrawingClassifier imports successfully')" || echo "❌ DrawingClassifier import failed"

# Test scripts import
python -c "import sys; sys.path.append('scripts'); import prepare_data; print('✅ prepare_data.py imports successfully')" || echo "❌ prepare_data.py import failed"

python -c "import sys; sys.path.append('scripts'); import train_model; print('✅ train_model.py imports successfully')" || echo "❌ train_model.py import failed"

echo ""
echo "📄 Checking documentation files..."
if [ -f "docs/API.md" ] && [ -f "docs/TRAINING.md" ] && [ -f "docs/DEPLOYMENT.md" ]; then
    echo "✅ All documentation files exist"
else
    echo "❌ Missing documentation files"
fi

echo ""
echo "🐳 Checking Docker configuration..."
if [ -f "docker/Dockerfile" ] && [ -f "docker/.dockerignore" ]; then
    echo "✅ Docker files exist"
else
    echo "❌ Missing Docker files"
fi

echo ""
echo "🎉 Project structure test completed!"
echo ""
echo "🚀 Ready to run:"
echo "   python scripts/prepare_data.py  # Prepare dataset"
echo "   python scripts/train_model.py   # Train model"
echo "   python app.py                   # Start API"
