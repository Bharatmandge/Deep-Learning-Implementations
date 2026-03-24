import cv2
import numpy as np

def enhance_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Image not loaded")

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)

    sharpened = cv2.filter2D(img, -1, kernel)

    return sharpened