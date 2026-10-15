from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_classifier(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: str = "cpu",
    save_fig_path: str | Path = "results/figures/confusion_matrix.png",
) -> Dict[str, Any]:
    """Evaluate trained classifier and save confusion matrix plot."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    labels_indices = list(range(len(class_names)))
    cm = confusion_matrix(all_targets, all_preds, labels=labels_indices)
    report = classification_report(
        all_targets, all_preds, labels=labels_indices, target_names=class_names, zero_division=0
    )

    # Save Confusion Matrix Figure
    import matplotlib.pyplot as plt
    out_path = Path(save_fig_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels_indices)
    ax.set_yticks(labels_indices)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"KITTI Crop Classification (Acc: {acc:.2%})")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    return {"accuracy": acc, "confusion_matrix": cm, "report": report}