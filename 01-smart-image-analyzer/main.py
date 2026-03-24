import matplotlib.pyplot as plt
import cv2
from classifier import classify_image
from detector import detect_objects
from enhancer import enhance_image

img_path = "images/image.png"

label = classify_image(img_path)
detected_img = detect_objects(img_path)
enhanced_img = enhance_image(img_path)

original_img = cv2.imread(img_path)
original_imgw = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.title("Original")
plt.imshow(original_imgw)

plt.subplot(1,3,2)
plt.title("Detection")
plt.imshow(detected_img)

plt.subplot(1,3,3)
plt.title("Enhanced")
plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))

plt.suptitle(f"Predicted Class ID: {label}")

plt.show()