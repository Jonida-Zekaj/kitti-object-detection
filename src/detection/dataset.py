from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from detection.dataset_adapter import KittiDetector, KittiObject


class KittiDetectionDataset(Dataset):
    def __init__(
        self,
        adapter: KittiDetector,
        grid_size: int = 13,
        num_classes: int = 5,
        target_size: Tuple[int, int] = (416, 416),
    ):
        self.adapter = adapter
        self.S = grid_size
        self.C = num_classes
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.adapter)

    def encode_target_grid(
        self, objects: List[KittiObject]
    ) -> torch.Tensor: #ground truth bounding boxes into a tensor of shape (S, S, 5 + C)

        target = torch.zeros((self.S, self.S, 5 + self.C), dtype=torch.float32)
        target_w, target_h = self.target_size

        for obj in objects:
            xmin, ymin, xmax, ymax = obj.bbox
            
            # Box center and dimensions relative to full image
            cx = (xmin + xmax) / 2.0 / target_w
            cy = (ymin + ymax) / 2.0 / target_h
            w = (xmax - xmin) / target_w
            h = (ymax - ymin) / target_h

            # Determine grid cell indices
            grid_i = int(cy * self.S)
            grid_j = int(cx * self.S)

            # Clamp to grid bounds
            grid_i = min(max(0, grid_i), self.S - 1)
            grid_j = min(max(0, grid_j), self.S - 1)

            # If no object assigned to cell yet, assign this object
            if target[grid_i, grid_j, 4] == 0:
                # Local offset within grid cell [0, 1]
                x_cell = cx * self.S - grid_j
                y_cell = cy * self.S - grid_i

                target[grid_i, grid_j, 0] = x_cell
                target[grid_i, grid_j, 1] = y_cell
                target[grid_i, grid_j, 2] = w
                target[grid_i, grid_j, 3] = h
                target[grid_i, grid_j, 4] = 1.0  # Objectness confidence
                
                # One-hot encode class
                if 0 <= obj.class_id < self.C:
                    target[grid_i, grid_j, 5 + obj.class_id] = 1.0

        return target

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_np, objects = self.adapter.load_sample(idx, resize=True)
        image_tensor = torch.tensor(image_np, dtype=torch.float32).permute(2, 0, 1) / 255.0
        target_tensor = self.encode_target_grid(objects)
        
        return image_tensor, target_tensor