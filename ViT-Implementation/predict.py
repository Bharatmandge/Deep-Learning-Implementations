import torch 
import numpy as np 
import torch.nn as nn 
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from models.vit_model import VisionTransformer 
from config import Config

def show_prediction():
    # 1 Setup 
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cifar 10 class Names 
    classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    # 2 Load the model
    model = VisionTransformer(cfg).to(device)
    try:
        model.load_state_dict(torch.load("vit_weights.pth", map_location=device))
        print("Successfully loaded model weights")
    except FileNotFoundError:
        print("MOdel could jot error ")
        return 
    
    model.eval()
    
    # 3. Load The TEst Dataset 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # 4 Pick a random image 
    idx = np.random.randint(len(test_data))
    image, actual_label = test_data[idx]
    
    # 5. Make prediction 
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        _, predicted_label = torch.max(output, 1)
        
    
    # 6. Visualize the output 
    img_display = image / 2 + 0.5
    img_display = np.transpose(img_display.numpy(), (1, 2, 0))
    
    actual_name = classes[actual_label]
    predicted_name = classes[predicted_label.item()]
    
    color = "green" if actual_name == predicted_name else "red"
    
    plt.figure(figsize=(4,4))
    plt.imshow(img_display)
    plt.title(f"AI Guessed: {predicted_name}\nActual: {actual_name}", color=color, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_prediction()