from backbone.resnet50.aptos_dataset import AptosDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models

# === Training script ===
if __name__ == '__main__':
    # paths and params
    video_dir = 'dataset/videos'
    ann_dir = 'dataset/annotations/APTOS_train-val_annotation.csv'
    # datasets and loaders
    train_ds = AptosDataset(video_dir, ann_dir, batch_size=16)
    train_loader = DataLoader(train_ds, batch_size=None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    # model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 35)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 3
    train_losses = []

    for epoch in range(num_epochs):
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

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}")

    # plots
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.show()
    torch.save(model.state_dict(), "backbone/resnet50/models/bresnet50_finetune_v1.pth")
