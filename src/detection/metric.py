import torch

# Let IoU = Intersection of Union
# Does the bbox predicted match the bbox ground truth and by how much
 
def calc_iou(box1: torch.Tensor, 
               box2: torch.Tensor) -> torch.Tensor:
    # Get coordinates of the intersection rectangle
    x1_inter = torch.max(box1[..., 0], box2[..., 0])
    y1_inter = torch.max(box1[..., 1], box2[..., 1])
    x2_inter = torch.min(box1[..., 2], box2[..., 2])
    y2_inter = torch.min(box1[..., 3], box2[..., 3])

    # Calculate intersection area
    inter_w = torch.clamp(x2_inter - x1_inter, min=0)
    inter_h = torch.clamp(y2_inter - y1_inter, min=0)
    inter_area = inter_w * inter_h

    # Calculate union area
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union_area = box1_area + box2_area - inter_area + 1e-6

    return inter_area / union_area

