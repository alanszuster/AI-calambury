# Docker Configuration

This directory contains Docker-related configuration files.

## Files

- [`Dockerfile`](Dockerfile) - Docker image configuration
- [`.dockerignore`](.dockerignore) - Files to exclude from Docker build context

## Usage

```bash
# Build Docker image
docker build -f docker/Dockerfile -t ai-drawing-classifier .

# Run container
docker run -p 5000:5000 ai-drawing-classifier
```

## Notes

- The Dockerfile is optimized for production deployment
- Uses multi-stage build for smaller image size
- Includes all necessary dependencies for TensorFlow
