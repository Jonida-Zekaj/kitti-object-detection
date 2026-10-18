# KITTI Object Detection and Tracking

A computer vision project for object detection and multi-object tracking using the [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/). The project is organized into classification, detection, tracking, and pipeline modules, with a command-line interface for running inference on individual images.

## Features

- **KITTI data handling:** Load and preprocess KITTI images and annotations.
- **Image classification:** PyTorch-based CNN classification components.
- **Object detection:** Grid-based neural network for predicting bounding boxes and object classes.
- **Non-Maximum Suppression (NMS):** Filter overlapping detections.
- **Multi-object tracking:** SORT-style tracking using bounding-box overlap and motion estimation.
- **Integrated pipeline:** Combine detection and tracking for frame-by-frame processing.
- **Hardware acceleration:** Automatically select Apple Silicon MPS, NVIDIA CUDA, or CPU.
- **Visualization:** Draw bounding boxes and tracking IDs on output images.

## Project Structure

```text
kitti-object-detection/
├── configs/
│   ├── default.yaml
│   ├── classification.yaml
│   ├── detection.yamltraining/
│   └── tracking.yaml
├── data/
│   ├── training/
│   └── testing/
├── results/
│   ├── checkpoints/
│   └── figures/
├── src/
│   ├── classification/
│   │   ├── classifier_model.py
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   └── train.py
│   ├── detection/
│   │   ├── dataset_adapter.py
│   │   ├── detector_model.py
│   │   ├── predict.py
│   │   └── train.py
│   ├── pipeline/
│   │   └── pipeline.py
│   ├── tracking/
│   │   ├── kalman_tracker.py
│   │   └── sort.py
│   ├── utils/
│   │   └── config.py
│   └── main.py
├── tests/
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or later
- PyTorch
- OpenCV
- NumPy
- SciPy
- PyYAML
- pytest (for testing)

Python 3.13 may work with compatible package versions. If an installed dependency does not support your Python version, use a compatible Python environment.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Jonida-Zekaj/kitti-object-detection
cd kitti-object-detection
```


### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file exists:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Otherwise, install the packages used by the project:

```bash
pip install torch torchvision numpy opencv-python scipy pyyaml pytest
```

### 4. Configure Python imports

The project uses imports such as `from detection...` and `from pipeline...`. Run commands from the repository root with `PYTHONPATH=src`.

For example:

```bash
PYTHONPATH=src python3 src/main.py
```

## Dataset Setup

Download the KITTI object detection dataset from the [official KITTI website](https://www.cvlibs.net/datasets/kitti/).

For object detection, organize the relevant files as follows:

```text
data/kitti/
└── training/
    ├── image_2/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    └── label_2/
        ├── 000000.txt
        ├── 000001.txt
        └── ...
```

- `image_2/` contains the RGB camera images.
- `label_2/` contains the corresponding object annotations.

Ensure that image and label filenames correspond. Follow the dataset adapter's expected directory structure if it differs from the layout above.

## Running the Pipeline

The command-line entry point is `src/main.py`.

### Initialize the pipeline

```bash
PYTHONPATH=src python3 src/main.py
```

This initializes the pipeline and reports the selected device.

### Process an image

```bash
PYTHONPATH=src python3 src/main.py \
  --image data/kitti/training/image_2/000000.png
```

### Specify model weights

```bash
PYTHONPATH=src python3 src/main.py \
  --image data/kitti/training/image_2/000000.png \
  --weights results/checkpoints/detector_init.pth
```

### Specify an output path

```bash
PYTHONPATH=src python3 src/main.py \
  --image data/kitti/training/image_2/000000.png \
  --weights results/checkpoints/detector_init.pth \
  --output results/figures/pipeline_output.png
```

The output image is saved at the specified path, with detected tracks visualized using bounding boxes and track IDs.

**Important:** The checkpoint must match the model architecture. A checkpoint whose final layer has 15 channels is not compatible with a model configured to produce 10 channels. Configure the model and decoder consistently before running inference.

The checkpoint must also contain trained weights for meaningful predictions. An initialization checkpoint alone does not guarantee useful detections.




## Model Training

The detection module contains the detector model and training entry point.

Before training, verify that:

- The dataset paths are correct.
- The annotation parser matches KITTI's label format.
- The detector's output shape matches the loss function.
- Bounding-box coordinate conventions are consistent.
- Checkpoints are saved with the architecture and parameters required for inference.

Run the training script using the options supported by your implementation. For example:

```bash
PYTHONPATH=src python3 src/detection/train.py
```

Check the script's arguments and configuration before running it, as training options may vary.

## Object Tracking

The tracking module associates detections between frames using bounding-box overlap and motion prediction.

The tracker configuration is stored in `configs/tracking.yaml`. Typical parameters include:

| Parameter | Description |
|---|---|
| `max_age` | Maximum number of frames a track can remain unmatched before removal |
| `min_hits` | Minimum number of detections required before a track is confirmed |
| `iou_threshold` | Minimum IoU used when matching detections to existing tracks |

Tracking is intended for sequential frames from a video or image sequence. Processing unrelated images independently will not provide meaningful persistent identities.

## Testing

Run the test suite from the repository root:

```bash
PYTHONPATH=src python3 -m pytest -v
```

To run a specific real-data test, if available:

```bash
PYTHONPATH=src python3 -m pytest -v -s tests/test_real_data.py
```

Tests that depend on KITTI data require the dataset to be present at the configured path.

## Configuration

Configuration files are located in `configs/`:

- `default.yaml` — general project settings.
- `classification.yaml` — classification settings.
- `detection.yaml` — object detection settings.
- `tracking.yaml` — tracking parameters.

Review the configuration files and update paths, thresholds, and model settings to match your environment.



## Future Improvements

- Train and evaluate the detector on KITTI annotations.
- Implement and validate decoding for all predicted boxes per grid cell.
- Evaluate detection quality with precision, recall, and mean Average Precision (mAP).
- Evaluate tracking with suitable multi-object tracking metrics.
- Add video input and annotated video output.
- Improve Kalman filtering and track association.
- Add reproducible training configurations and experiment logging.