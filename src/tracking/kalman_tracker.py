import numpy as np
from typing import Tuple


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    #bounding box [xmin, ymin, xmax, ymax] into observation vector z = [u, v, s, r]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    u = bbox[0] + w / 2.0
    v = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([u, v, s, r]).reshape((4, 1))


def z_to_bbox(x: np.ndarray) -> np.ndarray:
    # state vector x = [u, v, s, r, ...] back to [xmin, ymin, xmax, ymax]
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w if w > 0 else 0.0
    return np.array([
        x[0] - w / 2.0,
        x[1] - h / 2.0,
        x[0] + w / 2.0,
        x[1] + h / 2.0,
    ]).flatten()


class KalmanBoxTracker:
    # a SINGLE object bbox is tracked  across video frames using linear velocity
    
    count = 0

    def __init__(self, bbox: np.ndarray):
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        
        # State vector: [u, v, s, r, u_dot, v_dot, s_dot]
        self.x = np.zeros((7, 1))
        self.x[:4] = bbox_to_z(bbox)

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self) -> np.ndarray:
        # Updating center coordinates based on estimated velocity
        self.x[0] += self.x[4]
        self.x[1] += self.x[5]
        self.x[2] += self.x[6]

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return z_to_bbox(self.x)

    def update(self, bbox: np.ndarray) -> None:
        # updates state estimate with an observed bounding box measurement 
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        # Smooth positional updates
        z = bbox_to_z(bbox)
        alpha = 0.6
        self.x[4] = alpha * (z[0] - self.x[0]) + (1 - alpha) * self.x[4]
        self.x[5] = alpha * (z[1] - self.x[1]) + (1 - alpha) * self.x[5]
        self.x[:4] = z

    def get_state(self) -> np.ndarray:
        """Returns the current estimated bounding box [xmin, ymin, xmax, ymax]."""
        return z_to_bbox(self.x)