from backbone.resnet50.aptos_dataset import AptosDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, 100. * correct / total

# === Training script ===
if __name__ == '__main__':
    # paths and params
    video_dir = 'dataset/videos'
    ann_dir = 'dataset/annotations/APTOS_train-val_annotation.csv'
    # datasets and loaders
    train_ds = AptosDataset(video_dir, ann_dir, batch_size=16)
    train_loader = DataLoader(train_ds, batch_size=None)

    val_ds = AptosDataset(video_dir, ann_dir, split='val' ,batch_size=16)
    val_loader = DataLoader(val_ds, batch_size=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 35)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 3
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        sample_count = 0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            sample_count += inputs.size(0)
            if sample_count % 128 == 0:
                print(f"Sample {sample_count} - Loss: {loss.item():.4f}")
        
        epoch_train_loss = running_loss / sample_count
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "backbone/resnet50/models/bresnet50_finetune_v1_best.pth")
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        print("-" * 50)

    # Save final model
    torch.save(model.state_dict(), "backbone/resnet50/models/bresnet50_finetune_v1_final.pth")

    # plots
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Val Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.show()
