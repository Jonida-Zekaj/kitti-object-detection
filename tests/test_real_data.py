from pathlib import Path 
from detection.dataset_adapter import KittiDetector

adapter = KittiDetector(
    image_dir=Path("data/training/image_2"),
    
    label_dir=Path("data/training/label_2"),
)

print(f"Images found: {len(adapter)}")

if len(adapter) > 0:
    image, objects = adapter.load_sample(0)
    print(f"Image shape: {image.shape}")
    print(f"Objects found: {len(objects)}")

    for obj in objects:
        print(f"{obj.label}: bbox={obj.bbox}")
else:
    print("No PNG images found. Check your dataset directory.")

    # goal: check data loading and parsing of KITTI dataset