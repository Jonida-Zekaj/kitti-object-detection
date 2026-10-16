import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

from tracking.kalman_tracker import KalmanBoxTracker


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:

    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h

    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
        + 1e-6
    )
    return o


class SortTracker: # Multi-object tracker
    def __init__(self, max_age: int = 5, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, dets: np.ndarray) -> np.ndarray:
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 5))

        # Predict new positions of existing tracks
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:4] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)

        for t in reversed(to_del):
            self.trackers.pop(t)

        trks = trks[~np.isin(np.arange(len(trks)), to_del)]

        # Match detections to predicted tracks using Hungarian algorithm
        if len(trks) > 0 and len(dets) > 0:
            iou_matrix = iou_batch(dets[:, :4], trks[:, :4])
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            matched_indices = []
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c))

            unmatched_dets = [d for d in range(len(dets)) if d not in [m[0] for m in matched_indices]]
            unmatched_trks = [t for t in range(len(trks)) if t not in [m[1] for m in matched_indices]]
        else:
            matched_indices = []
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(trks)))

        # Update matched trackers
        for m in matched_indices:
            self.trackers[m[1]].update(dets[m[0], :4])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        # Output active track outputs
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or trk.age <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))