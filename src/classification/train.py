from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.config import load_config
from detection.dataset_adapter import KittiDetectionAdapter
from classification.dataset import extract_object_crops, KittiCropDataset
from classification.model import KittiCropClassifier
from classification.evaluate import evaluate_classifier


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_training(max_images: int = 500) -> None:
    base_cfg = load_config("configs/default.yaml")
    cls_cfg = load_config("configs/classification.yaml")

    classes = base_cfg["dataset"]["classes"]
    crop_size = tuple(cls_cfg["model"]["crop_size"])

    adapter = KittiDetectionAdapter(
        image_dir=base_cfg["paths"]["train_images"],
        label_dir=base_cfg["paths"]["train_labels"],
        classes=classes,
    )

    print(f"Extracting object crops from first {max_images} KITTI images...")
    crops, labels = extract_object_crops(
        adapter,
        crop_size=crop_size,
        max_images=max_images,
        max_samples_per_class=cls_cfg["training"]["max_samples_per_class"],
    )
    print(f"Extracted {len(labels)} object crops across {len(classes)} classes.")

    X_train, X_val, y_train, y_val = train_test_split(
        crops,
        labels,
        test_size=cls_cfg["training"]["val_split"],
        random_state=base_cfg["dataset"]["random_seed"],
        stratify=labels,
    )

    batch_size = cls_cfg["training"]["batch_size"]
    train_loader = DataLoader(KittiCropDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(KittiCropDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    device = get_device()
    print(f"Training on device: {device}")

    model = KittiCropClassifier(
        num_classes=len(classes),
        dropout=cls_cfg["model"]["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cls_cfg["training"]["learning_rate"])

    epochs = cls_cfg["training"]["epochs"]
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f}")

    # Save checkpoint
    ckpt_path = Path(cls_cfg["training"]["checkpoint_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model checkpoint to: {ckpt_path}")

    # Evaluate & save confusion matrix
    metrics = evaluate_classifier(model, val_loader, class_names=classes, device=device)
    print(f"\nValidation Accuracy: {metrics['accuracy']:.2%}")
    print(metrics["report"])


if __name__ == "__main__":
    run_training(max_images=500)