import torch 
import torch.nn as nn


class Detector(nn.Module):
    def __init__(self, num_classes: int = 5, grid_size: int = 13, num_boxes: int = 2):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        
        # Lightweight backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Detection Head
        # Output depth: (B * 5) + C  
        # where 5 corresponds to (x, y, w, h, confidence)
        
        out_channels = (self.B * 5) + self.C
        self.head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, out_channels, kernel_size=1) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # Resize features to S x S grid if not already
        features = nn.functional.adaptive_avg_pool2d(features, (self.S, self.S))
        out = self.head(features)
        
        # Reshape to (Batch, S, S, Output_Dim)
        out = out.permute(0, 2, 3, 1)
        return out