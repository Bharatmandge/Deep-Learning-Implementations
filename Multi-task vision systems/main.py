import argparse
import sys
import os
import cv2
import numpy as np
import torch

from pipeline.visualizer import Visualizer
from pipeline.segmentor import Segmentor
from pipeline.metrics import compute_all_metrics
from pipeline.detector import Detector

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi task vision pipeline : Detection + segmentation + metrics",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--yolo-size", type=str, default='n', choices=['n','s','m','l','x'])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="pipeline_output.png")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def load_ground_truth_mask(mask_path, target_shape):
    gt_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_raw is None:
        raise FileNotFoundError(f"Ground truth mask not found at: {mask_path}")
    h, w = target_shape
    gt_resized = cv2.resize(gt_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    gt_binary = (gt_resized > 127).astype(np.uint8)
    foreground_pct = gt_binary.mean() * 100
    print(f"[GT Mask] Loaded from {mask_path} - foreground : {foreground_pct:.1f}%")
    return gt_binary


def simulate_ground_truth_mask(pred_mask):
    kernel = np.ones((15, 15), np.uint8)
    # FIX 1: typo "interations" → "iterations"
    dilated = cv2.dilate(pred_mask, kernel, iterations=1)
    print("[GT Mask] No mask provided. Using simulated GT (demo only)")
    return dilated


def run_pipeline(args):
    print("\n" + "=" * 58)
    print("  MULTI-TASK VISION PIPELINE")
    print("  Detection + Segmentation + Metrics")
    print("=" * 58 + "\n")

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Config] Device         : {device}")
    print(f"[Config] YOLO size      : {args.yolo_size}")
    print(f"[Config] YOLO confidence: {args.conf}")
    print(f"[Config] Mask threshold : {args.threshold}")
    print(f"[Config] U-Net weights  : {args.weights or 'None (random)'}")
    print(f"[Config] Output path    : {args.output}\n")

    # STEP 1: Load image
    print("-" * 40)
    print("STEP 1 / 6  ->  Load Image")
    print("-" * 40)
    original = cv2.imread(args.image)
    if original is None:
        print(f"[ERROR] Could not load image: {args.image}")
        sys.exit(1)
    print(f"[Image] Loaded: {args.image}")
    print(f"[Image] Shape : {original.shape}  (H x W x Channels)")

    # STEP 2: Detection
    print("\n" + "-" * 40)
    print("STEP 2 / 6  ->  Object Detection (YOLOv8)")
    print("-" * 40)
    detector = Detector(model_size=args.yolo_size, confidence_threshold=args.conf)
    boxes, annotated_img, class_names, confidences = detector.detect(original)

    if len(boxes) == 0:
        print("[Pipeline] No detections found. Using full image as fallback.")
        h, w = original.shape[:2]
        boxes = np.array([[0, 0, w, h]], dtype=np.float32)
        class_names = ['full_image']
        confidences = [1.0]

    primary_box = detector.get_primary_box(boxes, strategy='largest')
    print(f"[Pipeline] Primary box selected: {primary_box}")

    # STEP 3+4: Crop + Segment
    print("\n" + "-" * 40)
    print("STEP 3 / 6  ->  Crop Detected Region")
    print("STEP 4 / 6  ->  U-Net Segmentation")
    print("-" * 40)

    # FIX 2: renamed variable from 'segmentor' to 'seg' — avoids shadowing the imported class
    seg = Segmentor(weights_path=args.weights, device=device)
    cropped_img, seg_mask = seg.segment_with_crop(original, primary_box)
    if args.threshold != 0.5:
        seg_mask = seg.segment(cropped_img, threshold=args.threshold)

    # STEP 5: Metrics
    print("\n" + "-" * 40)
    print("STEP 5 / 6  ->  Load GT Mask + Compute Metrics")
    print("-" * 40)
    crop_shape = (cropped_img.shape[0], cropped_img.shape[1])
    if args.mask:
        gt_mask = load_ground_truth_mask(args.mask, crop_shape)
    else:
        gt_mask = simulate_ground_truth_mask(seg_mask)
    metrics = compute_all_metrics(pred_mask=seg_mask, gt_mask=gt_mask)

    # STEP 6: Visualize
    print("\n" + "-" * 40)
    print("STEP 6 / 6  ->  Visualize")
    print("-" * 40)

    # FIX 3: renamed variable from 'visualizer' to 'viz' — avoids shadowing the imported class
    viz = Visualizer(save_path=args.output, dpi=150)
    viz.render(
        original_bgr  = original,
        annotated_bgr = annotated_img,
        cropped_bgr   = cropped_img,
        seg_mask      = seg_mask,
        metrics       = metrics,
        show          = not args.no_show
    )

    print("\n" + "=" * 58)
    print("  PIPELINE COMPLETE")
    print(f"  Detected objects  : {len(boxes)} ({', '.join(class_names)})")
    print(f"  IoU Score         : {metrics['iou']:.4f}")
    print(f"  Dice Score        : {metrics['dice']:.4f}")
    print(f"  Output saved to   : {args.output}")
    print("=" * 58 + "\n")

    return metrics


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args)