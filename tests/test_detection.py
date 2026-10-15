from pathlib import Path

from detection.dataset_adapter import (
    KittiDetector,
    KittiObject,
)


def test_parse_and_filter_labels(tmp_path: Path):
    label_file = tmp_path / "000000.txt"

    label_file.write_text(
        "Car 0.00 0 -1.57 600.0 100.0 800.0 300.0 "
        "1.48 1.60 3.69 1.24 1.47 8.41 -1.56\n"
        "DontCare -1 -1 -10 500.0 150.0 550.0 200.0 "
        "-1 -1 -1 -1000 -1000 -1000 -10\n",
        encoding="utf-8",
    )

    adapter = KittiDetector(
        image_dir=tmp_path,
        label_dir=tmp_path,
        classes=["Car", "Pedestrian", "Cyclist"],
    )

    objects = adapter.parse_label_file(label_file)

    assert len(objects) == 1
    assert objects[0].label == "Car"
    assert objects[0].class_id == 0
    assert objects[0].bbox == (600.0, 100.0, 800.0, 300.0)


def test_scale_bounding_boxes():
    obj = KittiObject(
        label="Car",
        class_id=0,
        bbox=(600.0, 100.0, 800.0, 300.0),
    )

    scaled = KittiDetector.scale_bboxes(
        [obj],
        orig_size=(1200, 400),
        target_size=(600, 200),
    )

    assert scaled[0].bbox == (300.0, 50.0, 400.0, 150.0)


def test_empty_label_file(tmp_path: Path):
    label_file = tmp_path / "empty.txt"
    label_file.write_text("", encoding="utf-8")

    adapter = KittiDetector(
        image_dir=tmp_path,
        label_dir=tmp_path,
    )

    assert adapter.parse_label_file(label_file) == []