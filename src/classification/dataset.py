from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np 
import torch 
from PIL import Image
from torch.utils.data import Dataset
from detection.dataset_adapter import KittiDetector


def extract_object_crops(
    adapter: KittiDetector,
    crop_size: Tuple[int, int] = (64, 64),
    max_images: Optional[int] = None,
    max_samples_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    # crops = [N, H, W, 3]
    # labels = [N]
    crops: List[np.ndarray] = []
    labels: List[int] = []
    class_counts: Dict[int, int] = {i: 0 for i in range(len(adapter.classes))}

    num_images = len(adapter) if max_images is None else min(len(adapter), max_images)

    for idx in range(num_images):
        img_np, objects = adapter.load_sample(idx, resize=False)
        h, w, _ = img_np.shape
        pil_img = Image.fromarray(img_np)

        for obj in objects:
            if max_samples_per_class and class_counts[obj.class_id] >= max_samples_per_class:
                continue

            xmin, ymin, xmax, ymax = obj.bbox
            # Clamp coordinates to image boundaries
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(w, int(xmax))
            ymax = min(h, int(ymax))

            # Skip invalid or tiny boxes (< 10x10 pixels)
            if (xmax - xmin) < 10 or (ymax - ymin) < 10:
                continue

            crop = pil_img.crop((xmin, ymin, xmax, ymax)).resize(crop_size)
            crop_arr = np.asarray(crop, dtype=np.float32) / 255.0

            crops.append(crop_arr)
            labels.append(obj.class_id)
            class_counts[obj.class_id] += 1

    if not crops:
        return np.empty((0, crop_size[1], crop_size[0], 3), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(crops), np.array(labels, dtype=np.int64)