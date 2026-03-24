import torch
from torchvision import models, transforms
from PIL import Image

model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def classify_image(img_path):
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)

    _, predicted = torch.max(outputs, 1)
    return predicted.item()