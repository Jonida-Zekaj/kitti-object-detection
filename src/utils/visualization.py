from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np

from detection.dataset_adapter import KittiObject


CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Car": (0, 200, 255),
    "Pedestrian": (255, 80, 80),
    "Cyclist": (80, 255, 120),
    "Van": (255, 180, 0),
    "Truck": (180, 100, 255),
}


def draw_kitti_objects(
    image: np.ndarray,
    objects: List[KittiObject],
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and class names on an RGB image."""
    canvas = image.copy()

    for obj in objects:
        xmin, ymin, xmax, ymax = map(int, obj.bbox)
        color = CLASS_COLORS.get(obj.label, (255, 255, 255))

        cv2.rectangle(
            canvas, (xmin, ymin), (xmax, ymax), color, thickness
        )

        cv2.putText(
            canvas,
            obj.label,
            (xmin, max(ymin - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return canvas


def save_comparison_figure(
    orig_img: np.ndarray,
    orig_objs: List[KittiObject],
    resized_img: np.ndarray,
    resized_objs: List[KittiObject],
    output_path: str | Path = (
        "results/figures/bbox_scaling_check.png"
    ),
) -> Path:
    """Save original and resized images side by side."""
    import matplotlib.pyplot as plt

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vis_orig = draw_kitti_objects(orig_img, orig_objs)
    vis_resized = draw_kitti_objects(resized_img, resized_objs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(vis_orig)
    axes[0].set_title(
        f"Original ({orig_img.shape[1]}x{orig_img.shape[0]})"
    )
    axes[0].axis("off")

    axes[1].imshow(vis_resized)
    axes[1].set_title(
        f"Resized & Scaled "
        f"({resized_img.shape[1]}x{resized_img.shape[0]})"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path