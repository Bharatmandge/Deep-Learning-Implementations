import numpy as np


def compute_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_dice(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    denom        = pred.sum() + gt.sum()

    if denom == 0:
        return 1.0
    return float(2 * intersection / denom)


# BUG 1 FIX: compute_precision was actually computing pixel accuracy (wrong logic)
# Renamed and fixed to actually compute precision correctly
def compute_pixel_accuracy(pred_mask, gt_mask):
    pred    = pred_mask.astype(bool)
    gt      = gt_mask.astype(bool)
    correct = (pred == gt).sum()
    total   = gt.size
    return float(correct / total)


# BUG 2 FIX: compute_recall was actually computing BOTH precision and recall
# but was named compute_recall — renamed to compute_precision_recall to match
# what compute_all_metrics calls: compute_precision_recall(pred_mask, gt_mask)
def compute_precision_recall(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return precision, recall


def compute_all_metrics(pred_mask, gt_mask):
    if pred_mask.shape != gt_mask.shape:
        # BUG 3 FIX: f-string was missing the opening 'f' on the first line
        # so {pred_mask.shape} was printing as literal text, not the value
        raise ValueError(
            f"Mask shape mismatch: pred={pred_mask.shape}, gt={gt_mask.shape}. "
            f"Resize one to match the other before computing metrics."
        )

    iou                = compute_iou(pred_mask, gt_mask)
    dice               = compute_dice(pred_mask, gt_mask)
    accuracy           = compute_pixel_accuracy(pred_mask, gt_mask)
    precision, recall  = compute_precision_recall(pred_mask, gt_mask)

    metrics = {
        "iou"           : round(iou, 4),
        "dice"          : round(dice, 4),
        "pixel_accuracy": round(accuracy, 4),
        "precision"     : round(precision, 4),
        "recall"        : round(recall, 4),
    }

    print("\n" + "=" * 42)
    print("  SEGMENTATION METRICS")
    print("=" * 42)
    print(f"  IoU (Jaccard)      : {metrics['iou']:.4f}")
    print(f"  Dice Score (F1)    : {metrics['dice']:.4f}")
    print(f"  Pixel Accuracy     : {metrics['pixel_accuracy']:.4f}")
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  Recall             : {metrics['recall']:.4f}")
    print("=" * 42)

    iou_val = metrics['iou']
    if iou_val >= 0.90:
        label = "EXCELLENT"
    elif iou_val >= 0.75:
        label = "GOOD"
    elif iou_val >= 0.50:
        label = "ACCEPTABLE"
    else:
        label = "POOR — check model weights or threshold"

    print(f"  Quality: {label}")
    print("=" * 42 + "\n")

    return metrics