import torch
import torch.optim as optim
from pathlib import Path

from detection.detector_model import Detector
from detection.dataset_adapter import KittiDetector
from utils.config import load_config

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def train_detector(epochs: int = 10):
    device = get_device()
    print(f"Initializing Full-Scene Detector training on {device}...")
    
    base_cfg = load_config("configs/default.yaml")
    classes = base_cfg["dataset"]["classes"]
    
    # Initialize Model
    model = Detector(num_classes=len(classes), grid_size=13, num_boxes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print("Model architecture built. Ready to implement custom detection loss function.")
    print("Next step: Constructing the collate_fn to map KittiObjects to the S x S grid tensor.")
    
    # Dummy training loop placeholder
    model.train()
    for epoch in range(1, epochs + 1):
        print(f"Epoch [{epoch}/{epochs}] - Ready for target tensor generation.")
        
    ckpt_path = Path("results/checkpoints/detector_init.pth")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved initial detector weights to {ckpt_path}")

if __name__ == "__main__":
    train_detector(epochs=2)

# run: PYTHONPATH=src python3 src/detection/train.py