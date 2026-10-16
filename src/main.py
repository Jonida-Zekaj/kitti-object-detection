import argparse
from pathlib import Path
import cv2
import torch

from utils.config import load_config
from pipeline import InferencePipeline


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def draw_tracks(frame: cv2.Mat, tracks: torch.Tensor | list | None) -> cv2.Mat: # Bboxes with IDs on the output frame

    out_frame = frame.copy()
    if tracks is None or len(tracks) == 0:
        return out_frame

    for trk in tracks:
        xmin, ymin, xmax, ymax, track_id = map(int, trk[:5])
        # Draw bounding box
        cv2.rectangle(out_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Draw track label
        label = f"ID: {track_id}"
        cv2.putText(
            out_frame,
            label,
            (xmin, max(15, ymin - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return out_frame


def run_cli():
    parser = argparse.ArgumentParser(description="KITTI Perception Engine (Detect + Track)")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--weights", type=str, default="results/checkpoints/detector_init.pth", help="Model weights path")
    parser.add_argument("--output", type=str, default="results/figures/inference_output.png", help="Output path")
    args = parser.parse_args()

    device = get_device()
    print(f"Initializing Inference Pipeline on {device}...")

    pipeline = InferencePipeline(weights_path=args.weights, device=device)

    if args.image and Path(args.image).exists():
        img = cv2.imread(args.image)
        tracks = pipeline.process_frame(img)
        annotated_img = draw_tracks(img, tracks)
        
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, annotated_img)
        print(f"Pipeline executed successfully. Processed frame saved to {args.output}")
    else:
        print("Pipeline dry-run completed. Specify --image <path> to generate visual inference outputs.")


if __name__ == "__main__":
    run_cli()

# run: PYTHONPATH=src python3 src/main.py