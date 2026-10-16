import torch 
from torchvision.ops import nms
from typing import List, Tuple 

def apply_nms(
    predictions: torch.Tensor, 
    conf_threshold: float = 0.5, 
    iou_threshold: float = 0.4) -> List[torch.Tensor]: # where nms = non-maximum supression

    kept_boxes = []
    
    for img_preds in predictions:
        # Filter by confidence threshold
        mask = img_preds[:, 4] > conf_threshold
        filtered_preds = img_preds[mask]
        
        if len(filtered_preds) == 0:
            kept_boxes.append(torch.empty((0, 6)))
            continue
            
        boxes = filtered_preds[:, :4]
        scores = filtered_preds[:, 4]
        class_ids = filtered_preds[:, 5]
        
        # Apply torchvision NMS
        keep_indices = nms(boxes, scores, iou_threshold)
        kept_boxes.append(filtered_preds[keep_indices])
        
    return kept_boxes