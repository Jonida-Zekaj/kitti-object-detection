import numpy as np
from tracking.sort import SortTracker


def test_sort_tracker_id_persistence():
    tracker = SortTracker(max_age=3, min_hits=1, iou_threshold=0.3)

    # Frame 1: Initial detection
    frame1_dets = np.array([[100.0, 100.0, 200.0, 200.0, 0.9]])
    tracks1 = tracker.update(frame1_dets)
    assert len(tracks1) == 1
    assigned_id = tracks1[0, 4]

    # Frame 2: Object shifts slightly across consecutive frames
    frame2_dets = np.array([[105.0, 105.0, 205.0, 205.0, 0.88]])
    tracks2 = tracker.update(frame2_dets)
    assert len(tracks2) == 1
    assert tracks2[0, 4] == assigned_id  # Track ID must remain continuous