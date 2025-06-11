import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from phase_recognition.dataset import OphNetFeatureDataset
from phase_recognition.mstcn import MultiStageModel
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm

# Cấu hình training
config = {
    'num_stages': 2,
    'num_layers': 8,
    'num_f_maps': 32,
    'dim': 2048,
    'num_classes': 35,
    'causal_conv': True,
    'batch_size': 16,
    'num_epochs': 100,
    'learning_rate': 0.0005,
    'seq_len': 256,
    'patience': 5  # Early stopping patience
}

# Thiết lập device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Tạo datasets và dataloaders
train_dataset = OphNetFeatureDataset(
    annotation_csv="dataset/annotations/image_annotations_with_features.csv",
    split="train",
    seq_len=config['seq_len']
)

val_dataset = OphNetFeatureDataset(
    annotation_csv="dataset/annotations/image_annotations_with_features.csv", 
    split="val",
    seq_len=config['seq_len']
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False
)

# Khởi tạo model
model = MultiStageModel(
    num_stages=config['num_stages'],
    num_layers=config['num_layers'],
    num_f_maps=config['num_f_maps'],
    dim=config['dim'],
    num_classes=config['num_classes'],
    causal_conv=config['causal_conv']
).to(device)

# Loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

def calculate_accuracy(predictions, targets):
    """Tính accuracy cho từng phase"""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    accuracies = {}
    for phase in range(config['num_classes']):
        mask = targets == phase
        if np.sum(mask) > 0:
            acc = accuracy_score(targets[mask], predictions[mask])
            accuracies[f'phase_{phase}_acc'] = acc
    
    accuracies['overall_acc'] = accuracy_score(targets, predictions)
    return accuracies

def visualize_predictions(epoch, predictions, targets):
    """Vẽ đồ thị predictions và ground truth"""
    plt.figure(figsize=(15, 5))
    plt.plot(predictions, label='Predictions')
    plt.plot(targets, label='Ground Truth')
    plt.legend()
    plt.title(f'Predictions vs Ground Truth - Epoch {epoch}')
    plt.savefig(f'visualizations/epoch_{epoch}.png')
    plt.close()

# Tạo thư mục lưu visualizations
os.makedirs('visualizations', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Training loop
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(config['num_epochs']):
    model.train()
    train_losses = []
    train_accuracies = []
    
    # Training loop
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
    for batch_idx, (features, labels) in enumerate(progress_bar):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(features)  # [num_stages, batch_size, num_classes, seq_len]
        
        # MS-TCN uses multiple stages, we calculate loss for all stages
        loss = 0
        for stage_pred in predictions:
            stage_pred = stage_pred.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_classes]
            loss += criterion(stage_pred.view(-1, config['num_classes']), labels.view(-1))
        
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        # Calculate accuracy using last stage predictions
        final_predictions = torch.argmax(predictions[-1], dim=1)
        batch_accuracies = calculate_accuracy(final_predictions.view(-1), labels.view(-1))
        train_accuracies.append(batch_accuracies['overall_acc'])
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{batch_accuracies["overall_acc"]:.4f}'
        })
    
    # Validation phase
    model.eval()
    val_losses = []
    val_accuracies = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            predictions = model(features)
            
            # Calculate validation loss
            val_loss = 0
            for stage_pred in predictions:
                stage_pred = stage_pred.transpose(1, 2).contiguous()
                val_loss += criterion(stage_pred.view(-1, config['num_classes']), labels.view(-1))
            
            val_losses.append(val_loss.item())
            
            # Store predictions and targets for visualization
            final_predictions = torch.argmax(predictions[-1], dim=1)
            all_predictions.extend(final_predictions.view(-1).cpu().numpy())
            all_targets.extend(labels.view(-1).cpu().numpy())
            
            # Calculate accuracy
            batch_accuracies = calculate_accuracy(final_predictions.view(-1), labels.view(-1))
            val_accuracies.append(batch_accuracies['overall_acc'])
    
    # Calculate epoch metrics
    epoch_train_loss = np.mean(train_losses)
    epoch_train_acc = np.mean(train_accuracies)
    epoch_val_loss = np.mean(val_losses)
    epoch_val_acc = np.mean(val_accuracies)
    
    # Visualize predictions
    if (epoch + 1) % 5 == 0:  # Visualize every 5 epochs
        visualize_predictions(epoch + 1, all_predictions, all_targets)
    
    # Early stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': epoch_val_loss,
        }, 'phase_recognition/models/mstcn_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= config['patience']:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    print(f'Epoch {epoch + 1}/{config["num_epochs"]}:')
    print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
    print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')