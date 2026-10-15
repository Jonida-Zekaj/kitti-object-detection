from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image


@dataclass 
# One 2D object annotation
class KittiObject: 
    label:str
    class_id: int
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    truncated: float= 0.0
    occluded: int = 0

# Class for loading images with their respective bboxes
class KittiDetector:
    def __init__(  
        self,
        image_dir: str | Path,
        label_dir: Optional[str | Path] = None,
        target_size: Tuple[int, int] = (416, 416),
        classes: Optional[List[str]] = None,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.target_size = target_size  # (width, height)

        self.classes = classes or [
            "Car", "Pedestrian", "Cyclist", "Van", "Truck"
        ]

        self.class_to_id = {
            name: idx for idx, name in enumerate(self.classes)
        }

        if self.image_dir.exists():
            self.image_files = sorted(self.image_dir.glob("*.png"))
        else:
            self.image_files = []

    def __len__(self) -> int:
        return len(self.image_files)

    def parse_label_file(
        self, label_path: str | Path
    ) -> List[KittiObject]:
        """Read labels and keep only configured classes."""
        path = Path(label_path)

        if not path.exists():
            return []

        objects = []

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()

                if len(parts) < 8:
                    continue

                label = parts[0]

                if label not in self.class_to_id:
                    continue

                try:
                    truncated = float(parts[1])
                    occluded = int(float(parts[2]))
                    xmin, ymin, xmax, ymax = map(
                        float, parts[4:8]
                    )
                except ValueError:
                    continue

                objects.append(
                    KittiObject(
                        label=label,
                        class_id=self.class_to_id[label],
                        bbox=(xmin, ymin, xmax, ymax),
                        truncated=truncated,
                        occluded=occluded,
                    )
                )

        return objects

    @staticmethod

# if the image was resized its bounding boxes MUST also be scaled accordingly

    def scale_bboxes(
        objects: List[KittiObject],
        orig_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> List[KittiObject]:

        
        orig_w, orig_h = orig_size
        target_w, target_h = target_size

        if orig_w <= 0 or orig_h <= 0:
            raise ValueError("Original image dimensions must be POSITIVE!")

        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

        scaled_objects = []

        for obj in objects:
            xmin, ymin, xmax, ymax = obj.bbox

            scaled_bbox = (
                xmin * scale_x, # format: xmin x * (416 / 1242)
                ymin * scale_y,
                xmax * scale_x,
                ymax * scale_y,
            )

            scaled_objects.append(
                KittiObject(
                    label=obj.label,
                    class_id=obj.class_id,
                    bbox=scaled_bbox,
                    truncated=obj.truncated,
                    occluded=obj.occluded,
                )
            )

        return scaled_objects

    def load_sample(
        self, idx: int, resize: bool = True
    ) -> Tuple[np.ndarray, List[KittiObject]]:
        #Load one image and its annotations
        
        img_path = self.image_files[idx]

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            orig_size = img.size

            if resize:
                img = img.resize(self.target_size)

            image_np = np.array(img)

        objects = []

        if self.label_dir is not None:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            objects = self.parse_label_file(label_path)

            if resize:
                objects = self.scale_bboxes(
                    objects, orig_size, self.target_size
                )

        return image_np, objects
