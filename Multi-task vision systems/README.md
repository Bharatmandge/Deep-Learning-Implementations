# Multi-Task Vision Pipeline

## Project Structure

```
vision_pipeline/
│
├── model/
│   └── unet.py           ← U-Net architecture (Encoder-Decoder)
│
├── pipeline/
│   ├── detector.py       ← YOLOv8 detection module
│   ├── segmentor.py      ← U-Net segmentation module
│   ├── metrics.py        ← IoU & Dice computation
│   └── visualizer.py     ← Final figure drawing
│
├── main.py               ← RUN THIS — ties everything together
└── README.md
```

## How to Run

```bash
pip install ultralytics torch torchvision opencv-python matplotlib pillow

python main.py --image your_image.jpg
# With ground truth mask:
python main.py --image your_image.jpg --mask your_mask.png
```