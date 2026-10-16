from pathlib import Path
from typing import List
import torch
import numpy as np
import cv2

from detection.detector_model import Detector
from detection.predict import apply_nms
from tracking.sort import SortTracker


class Pipeline:

    def __init__(
        self,
        weights_path: str | Path,
        num_classes: int = 5,
        grid_size: int = 13,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        self.model = Detector(num_classes=num_classes, grid_size=grid_size).to(self.device)
        if Path(weights_path).exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        self.tracker = SortTracker(max_age=5, min_hits=2, iou_threshold=0.3)

    @torch.no_grad()
    def process_frame(self, image_np: np.ndarray) -> np.ndarray:
        orig_h, orig_w = image_np.shape[:2]

        img_resized = cv2.resize(image_np, (416, 416))
        tensor_in = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor_in = tensor_in.to(self.device)

        preds = self.model(tensor_in)  # Shape: (1, S, S, Out_Dim)

        raw_boxes = []
        S = preds.size(1)
        stride_x = orig_w / float(S)
        stride_y = orig_h / float(S)

        for i in range(S):
            for j in range(S):
                cell = preds[0, i, j]
                conf = torch.sigmoid(cell[4]).item()
                if conf > self.conf_threshold:
                    x_cell = torch.sigmoid(cell[0]).item()
                    y_cell = torch.sigmoid(cell[1]).item()
                    w_norm = torch.sigmoid(cell[2]).item()
                    h_norm = torch.sigmoid(cell[3]).item()

                    # Correct grid coordinate projection
                    cx = (j + x_cell) * stride_x
                    cy = (i + y_cell) * stride_y
                    w = w_norm * orig_w
                    h = h_norm * orig_h

                    xmin = max(0.0, cx - w / 2.0)
                    ymin = max(0.0, cy - h / 2.0)
                    xmax = min(float(orig_w), cx + w / 2.0)
                    ymax = min(float(orig_h), cy + h / 2.0)

                    class_id = torch.argmax(cell[5:]).item()
                    raw_boxes.append([xmin, ymin, xmax, ymax, conf, class_id])

        if not raw_boxes:
            return self.tracker.update(np.empty((0, 5)))

        pred_tensor = torch.tensor(raw_boxes).unsqueeze(0)
        nms_results = apply_nms(
            pred_tensor, conf_threshold=self.conf_threshold, iou_threshold=self.iou_threshold
        )[0]

        if len(nms_results) == 0:
            return self.tracker.update(np.empty((0, 5)))

        dets_for_sort = nms_results[:, :5].cpu().numpy()
        tracks = self.tracker.update(dets_for_sort)
        return tracks