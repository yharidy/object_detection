# Object Detection with WAYM Dataset

A PyTorch-based 2D/3D object detection framework built for the WAYMO dataset.

## Project Structure

```
object_detection/
├── .devcontainer/          # DevContainer configuration
│   ├── devcontainer.json   # DevContainer settings
│   └── Dockerfile          # Container image definition
├── src/                    # Source code
│   ├── models/             # Model architectures
│   ├── data/               # Data loading utilities
│   ├── utils/              # Utility functions
│   └── train.py            # Training script
├── data/                   # Data directory (git-ignored)
│   ├── raw/                # Raw dataset files
│   └── processed/          # Processed dataset files
├── models/                 # Model checkpoints and weights
│   ├── checkpoints/        # Training checkpoints
│   └── pretrained/         # Pretrained weights
├── notebooks/              # Jupyter notebooks for exploration
├── config/                 # Configuration files
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

### Using DevContainer (Recommended)

1. Open the project in VS Code
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) and select "Dev Containers: Reopen in Container"
3. Wait for the container to build and dependencies to install

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

The WAYMO dataset should be placed in the `data/raw/` directory. Follow WAYMO's instructions for downloading and organizing the data.

## Quick Start

```python
# Example: Training a model (to be implemented)
from src.models import ObjectDetector
from src.data import WAYMDataLoader

# Initialize data loader
train_loader = WAYMDataLoader.get_loader('data/raw', split='train')

# Initialize model
model = ObjectDetector(model_type='yolov8')  # or custom model

# Training loop (to be implemented)
```

## Features

- [x] DevContainer setup with PyTorch
- [ ] Data loading utilities for WAYM dataset
- [ ] Model architectures (YOLOv8, custom 2D/3D detectors)
- [ ] Training pipeline with logging
- [ ] Evaluation metrics
- [ ] Model checkpointing and resuming
- [ ] Visualization utilities
- [ ] Inference script

## Dependencies

- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision utilities
- **OpenCV**: Image processing
- **NumPy/Pandas**: Data processing
- **Matplotlib/Seaborn**: Visualization
- **Jupyter**: Interactive development
- **TensorBoard**: Training monitoring

See `requirements.txt` for complete dependency list.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with flake8
flake8 src/ tests/
```

### Launching Jupyter

```bash
jupyter lab notebooks/
```

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and commit: `git commit -am 'Add feature'`
3. Push to the branch: `git push origin feature/your-feature`
4. Submit a pull request

## License

[Add your license here]

## References

- [WAYM Dataset](https://example.com/waym)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
