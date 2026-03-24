import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from model.unet import load_unet


class Segmentor:
    INPUT_SIZE = 256

    def __init__(self, weights_path=None, device='cpu'):
        # BUG 1 FIX: was 'weight_path' (missing 's') — must match main.py call
        self.device = device
        self.model = load_unet(weights_path).to(device)

        self.transform = transforms.Compose([
            # BUG 2 FIX: was 'self.Input_SIZE' — wrong capital I, Python is case-sensitive
            transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        print(f"[Segmentor] Ready on device: {device}")

    def preprocess(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        tensor = self.transform(pil_image)
        tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def postprocess(self, raw_output, original_size, threshold=0.5):
        prob_map = raw_output.squeeze().cpu().numpy()
        binary_mask_256 = (prob_map > threshold).astype(np.uint8)
        orig_h, orig_w = original_size
        binary_mask = cv2.resize(
            binary_mask_256,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST
        )
        return binary_mask

    def segment(self, image_bgr, threshold=0.5):
        original_size = (image_bgr.shape[0], image_bgr.shape[1])
        tensor = self.preprocess(image_bgr)
        with torch.no_grad():
            raw_output = self.model(tensor)
        mask = self.postprocess(raw_output, original_size, threshold)
        foreground_pct = mask.mean() * 100
        print(f"[Segmentor] Mask generated — foreground coverage: {foreground_pct:.1f}%")
        return mask

    def segment_with_crop(self, full_image_bgr, box):
        # BUG 3 FIX: YOLO returns float coords — cast to int before numpy slicing
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(full_image_bgr.shape[1], int(box[2]))
        y2 = min(full_image_bgr.shape[0], int(box[3]))

        cropped = full_image_bgr[y1:y2, x1:x2]

        if cropped.size == 0:
            raise ValueError(f"Empty crop from box {box} — check detection output")

        print(f"[Segmentor] Cropped region: {cropped.shape} from box [{x1},{y1},{x2},{y2}]")
        mask = self.segment(cropped)
        return cropped, mask