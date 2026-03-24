import cv2
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, save_path="pipeline_output.png", dpi=150):
        self.save_path = save_path
        self.dpi = dpi

    def _bgr_to_rgb(self, image_bgr):
        # BUG 1 FIX: cv2.cvtcolor → cv2.cvtColor (capital T and C)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    def _build_overlay(self, cropped_bgr, mask):
        base_rgb = self._bgr_to_rgb(cropped_bgr).copy().astype(np.float32)
        green = np.zeros_like(base_rgb)
        green[:, :, 1] = 255.0

        # BUG 2 FIX: axis=-1 not axis=1
        # axis=1 stacks along width dimension → wrong shape
        # axis=-1 stacks along channel dimension → correct (H, W, 3)
        mask_3ch = np.stack([mask, mask, mask], axis=-1).astype(bool)

        overlay = np.where(
            mask_3ch,
            base_rgb * 0.5 + green * 0.5,
            base_rgb
        )
        return overlay.astype(np.uint8)

    def _draw_metrics_panel(self, ax, metrics):
        ax.axis('off')
        iou  = metrics.get('iou', 0.0)
        dice = metrics.get('dice', 0.0)
        acc  = metrics.get('pixel_accuracy', 0.0)
        prec = metrics.get('precision', 0.0)
        rec  = metrics.get('recall', 0.0)

        if iou >= 0.75:
            quality_color = 'green'
            quality_label = 'GOOD'
        elif iou >= 0.50:
            quality_color = 'orange'
            quality_label = 'ACCEPTABLE'
        else:
            quality_color = 'red'
            quality_label = 'POOR'

        text = (
            f"IoU            : {iou:.4f}\n"
            f"Dice Score (F1): {dice:.4f}\n"
            f"Pixel Accuracy : {acc:.4f}\n"
            f"Precision      : {prec:.4f}\n"
            f"Recall         : {rec:.4f}\n"
            f"Quality        : {quality_label}"
        )
        ax.text(
            0.5, 0.55, text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='center',
            horizontalalignment='center',
            fontfamily='monospace',
            color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1',
                      edgecolor='#bdc3c7', linewidth=1.5)
        )
        ax.text(
            0.5, 0.12, f"  {quality_label}  ",
            transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            verticalalignment='center',
            horizontalalignment='center',
            color='white',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=quality_color, edgecolor='none')
        )

    def render(self, original_bgr, annotated_bgr, cropped_bgr, seg_mask, metrics, show=True):
        # BUG 3 FIX: parameter was 'annonated_bgr' (double n, missing t)
        # main.py calls render(annotated_bgr=...) so spelling must match exactly
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.patch.set_facecolor('#1a1a2e')
        fig.suptitle(
            "Multi-Task Vision Pipeline: Detection + Segmentation + Metrics",
            fontsize=14, fontweight='bold', color='white', y=0.98
        )

        panel_titles = [
            "1. Original Image",
            "2. Object Detection (YOLOv8)",
            "3. Cropped Region",
            "4. Segmentation Mask (U-Net)",
            "5. Segmentation Overlay",
            "6. Evaluation Metrics",
        ]

        axes[0, 0].imshow(self._bgr_to_rgb(original_bgr))
        axes[0, 1].imshow(self._bgr_to_rgb(annotated_bgr))
        axes[0, 2].imshow(self._bgr_to_rgb(cropped_bgr))
        axes[1, 0].imshow(seg_mask, cmap='gray', vmin=0, vmax=1)

        overlay = self._build_overlay(cropped_bgr, seg_mask)
        axes[1, 1].imshow(overlay)

        self._draw_metrics_panel(axes[1, 2], metrics)

        for i, ax in enumerate(axes.flat):
            ax.set_title(panel_titles[i], fontsize=10, color='white',
                         pad=6, fontweight='500')
            if i < 5:
                ax.axis('off')
            ax.set_facecolor('#16213e')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.save_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"[Visualizer] Figure saved -> {self.save_path}")

        if show:
            plt.show()

        plt.close(fig)