from pathlib import Path
from detection.dataset import KittiDetectionDataset
from detection.dataset_adapter import KittiDetector, KittiObject

def test_grid_target_encoding(tmp_path: Path):

    adapter = KittiDetector(
        image_dir=tmp_path,
        label_dir=tmp_path,
        classes=["Car", "Pedestrian", "Cyclist"],
    )
    dataset = KittiDetectionDataset(adapter=adapter, grid_size=13, num_classes=3)

    # Mock object at center of image (416x416 -> cx=208, cy=208 -> grid cell (6, 6))
    mock_obj = KittiObject(label="Car", class_id=0, bbox=(150.0, 150.0, 266.0, 266.0))
    target = dataset.encode_target_grid([mock_obj])

    assert target.shape == (13, 13, 8)  # 5 + 3 classes
    assert target[6, 6, 4] == 1.0       # Objectness set
    assert target[6, 6, 5] == 1.0       # One-hot class set 