from pathlib import Path
import numpy as np
from PIL import Image
import torch

from detection.dataset_adapter import KittiDetector
from classification.dataset import extract_object_crops, KittiCropDataset
from classification.classifier_model import KittiCropClassifier


def test_cnn_forward_shape():
    model = KittiCropClassifier(num_classes=5, dropout=0.4)
    dummy_input = torch.randn(8, 3, 64, 64)
    logits = model(dummy_input)
    assert logits.shape == (8, 5)


def test_extract_object_crops(tmp_path: Path):
    # Create a dummy 200x200 RGB image and matching label file
    img = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))
    img.save(tmp_path / "000000.png")

    label_file = tmp_path / "000000.txt"
    label_file.write_text(
        "Car 0.0 0 -1.5 20.0 30.0 100.0 120.0 1.5 1.6 3.7 1.0 1.0 8.0 -1.5\n"
        "Pedestrian 0.0 0 -1.5 110.0 40.0 150.0 160.0 1.7 0.6 0.8 2.0 1.0 5.0 -1.5\n",
        encoding="utf-8",
    )

    adapter = KittiDetector(
        image_dir=tmp_path,
        label_dir=tmp_path,
        classes=["Car", "Pedestrian", "Cyclist"],
    )

    crops, labels = extract_object_crops(adapter, crop_size=(64, 64))
    assert crops.shape == (2, 64, 64, 3)
    assert list(labels) == [0, 1]

    dataset = KittiCropDataset(crops, labels)
    sample_x, sample_y = dataset[0]
    assert sample_x.shape == (3, 64, 64)
    assert sample_y.item() == 0