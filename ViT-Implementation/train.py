import torch 
import torch.nn as nn 
import torch.optim as optim 
from config import Config
from data.dataset import get_dataloaders 
from models.vit_model import VisionTransformer

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    cfg = Config()
    train_loader, test_loader = get_dataloaders(cfg.batch_size)
    
    model = VisionTransformer(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    # This is the main training loop
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc:.2f}%") 
        
    # --- FIXED SECTION ---
    # We are still INSIDE the train() function here.
    # The 'for' loop has finished, so we save the model now.
    torch.save(model.state_dict(), "vit_weights.pth")
    print("Model saved to vit_weights.pth!")
    # ---------------------

# This tells Python to run the train() function when you execute the script
if __name__ == "__main__":
    train()