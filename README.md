# Object Detection Research Platform

A research-driven PyTorch project for building, training, and evaluating camera-based object detectors on the [Waymo Open Dataset](https://waymo.com/open/).

The project is designed as an incremental model-development platform: establish a reliable data and training pipeline, implement increasingly capable detection architectures, and measure how model choices affect real-world detection performance. The current focus is a from-scratch implementation of **Deformable DETR**.

## What This Project Demonstrates

- PyTorch model development and training workflows
- Transformer architecture design and implementation
- CNN feature extraction and multi-scale vision representations
- Hungarian matching and DETR classification, L1, and GIoU losses
- Waymo Open Dataset v2 loading from local storage or Google Cloud Storage
- Variable-length object-detection targets with custom batching
- Checkpointing and resume training from Google Drive in Colab
- TensorBoard loss, validation, and prediction visualization
- Unit testing, typed interfaces, documentation, and reproducible experiments
- Dev Container-based development with VS Code and Pylance

## Current Model: Deformable DETR

The main model is based on:

> Zhu et al., [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)

The implementation combines:

```text
Waymo camera image
        |
        v
ResNet-18 multi-scale backbone
        |
        v
2D sine positional encoding + padding masks
        |
        v
Flattened multi-scale feature sequence
        |
        v
Deformable transformer encoder
        |
        v
Learned object queries + reference points
        |
        v
Deformable transformer decoder
        |
        +--> class logits, including no-object
        +--> normalized cxcywh box predictions
```

Unlike dense attention, deformable attention predicts a small number of sampling offsets around reference points and gathers features from selected locations across multiple resolutions. The implementation uses memory-bounded query processing and direct bilinear sampling so encoder self-attention remains practical on Colab GPUs.

The architecture is documented in detail in [src/models/detr/README.md](src/models/detr/README.md).

## Training And Evaluation

The training pipeline currently supports camera-based 2D detection:

1. Waymo v2 Parquet data is loaded through source-specific segment and frame abstractions.
2. Images and labels are resized to a common training resolution.
3. Ground-truth boxes are represented internally as pixel-space `cxcywh`.
4. The model predicts normalized `cxcywh` boxes and class logits.
5. Per-image Hungarian matching assigns queries to ground-truth objects.
6. DETR loss combines classification, L1 box regression, and GIoU terms.
7. Validation is kept separate from optimization and evaluated after each epoch.
8. Detection KPIs include precision, recall/TP rate, F1, AP50, AP75, and mAP across IoU thresholds from 0.50 to 0.95.

The reusable metric implementation is in [src/utils/metrics.py](src/utils/metrics.py). It performs class-aware, one-to-one IoU matching and evaluates confidence-ranked predictions for AP and mAP.

## Colab Training Workflow

For long-running experiments, use the checkpointed notebook:

[Open the Deformable DETR Colab notebook](https://colab.research.google.com/github/yharidy/object_detection/blob/main/notebooks/train_deformable_detr_checkpointed_colab.ipynb)

The notebook supports:

- Google Drive checkpoint storage
- automatic resume from `latest.pt`
- historical epoch checkpoints
- configurable checkpoint version directories such as `v1`, `v2`, and `v3`
- automatic mixed precision on CUDA
- TensorBoard scalar metrics and prediction images
- separate Waymo `training` and `validation` partitions

Example configuration:

```python
CHECKPOINT_DIR = "/content/drive/MyDrive/deformable_detr/checkpoints/v1"
RESUME = True
TRAIN_NUM_SEGMENTS = 10
VAL_NUM_SEGMENTS = -1
```

The official `testing` partition is held back for final evaluation after model and hyperparameters have been selected using validation data.

## Repository Layout

```text
object_detection/
├── .devcontainer/                 # Reproducible VS Code development environment
├── config/                        # Configuration documentation and examples
├── notebooks/                     # Dataset exploration and Colab experiments
├── scripts/                       # Training, visualization, and data utilities
├── src/
│   ├── datasets/                  # Dataset adapters, transforms, and collation
│   ├── domain/                    # Camera, frame, box, label, and calibration models
│   ├── models/
│   │   ├── detr/                  # Deformable DETR implementation
│   │   └── faster_rcnn/            # Pipeline placeholder and comparison path
│   ├── sources/waymo/              # Waymo v2 storage and frame loading
│   ├── utils/                     # Box operations, metrics, logging, and helpers
│   └── viz/                       # Projection and visualization utilities
├── tests/                         # Unit and component tests
├── requirements.txt               # Pinned Python dependencies
└── pyproject.toml                 # Python package configuration
```

Faster R-CNN is retained as an early pipeline-validation placeholder. The research focus is currently Deformable DETR; additional architectures can be added incrementally once their training and evaluation behavior can be compared on the same data pipeline.

## Development Environment

The repository includes a Dev Container configuration for consistent development in VS Code. It provisions Python tooling, PyTorch dependencies, Pylance, formatting, linting, testing, Jupyter, and GitHub CLI support.

### Dev Container

1. Open the repository in VS Code.
2. Run **Dev Containers: Reopen in Container**.
3. Let the post-create setup install the project dependencies.

### Local installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The requirements are maintained for modern Python environments, including Colab's Python 3.13 runtime.

## Testing And Quality Checks

Run the test suite with:

```bash
pytest -q
```

The tests cover backbone outputs, positional encoding, feature flattening, deformable attention, encoder/decoder behavior, matching, DETR loss, and detection metrics.

Useful development commands:

```bash
python -m py_compile src/models/detr/*.py
git diff --check
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

## Research Direction

The project is intentionally developed as an iterative investigation rather than a single fixed model release. The next stages are:

- complete longer Deformable DETR training runs
- compare validation metrics across architecture and hyperparameter versions
- add stronger evaluation and error-analysis visualizations
- establish reproducible experiment records and selected checkpoints
- implement and compare additional detection architectures on the shared Waymo pipeline

The goal is to connect model theory with working engineering: understand the architecture, implement the critical components, train it on a realistic autonomous-driving dataset, and evaluate its behavior with defensible metrics.

## References

- [Waymo Open Dataset](https://waymo.com/open/)
- [Deformable DETR](https://arxiv.org/abs/2010.04159)
- [DETR](https://arxiv.org/abs/2005.12872)
- [PyTorch](https://pytorch.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
