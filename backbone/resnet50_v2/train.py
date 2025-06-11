import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T
from sklearn.metrics import classification_report
from backbone.resnet50_v2.dataset import PhaseImageDataset
from backbone.resnet50_v2.resnet50_finetune import PhaseResNet50Model
from backbone.resnet50_v2.utils import compute_class_weights

def worker_init_fn(worker_id):
    """Initialize random seed for DataLoader workers."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)

def train_one_epoch(model, loader, criterion, optimizer, device, log_every=128):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    running_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.view(-1).to(device)

        optimizer.zero_grad()
        _, logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_samples += inputs.size(0)
        if running_samples % log_every == 0:
            avg = running_loss / running_samples
            print(f"  [Train] processed {running_samples} samples, avg loss = {avg:.4f}")

    epoch_loss = running_loss / running_samples
    return epoch_loss

def validate(model, loader, criterion, device):
    """Validate the model on the validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.view(-1).to(device)

            _, logits = model(inputs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:\n", classification_report(all_labels, all_preds, zero_division=0))
    return running_loss / total, 100.0 * correct / total

def setup_device():
    """Set up the computation device (CUDA, MPS, or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available()
                         else "mps" if torch.backends.mps.is_available()
                         else "cpu")
    print("Using device:", device)
    return device

def setup_datasets_and_loaders(image_dir, ann_file, split_train, split_val, batch_size, num_workers):
    """Initialize datasets and data loaders with augmentation."""
    train_transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ConvertImageDtype(torch.float),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = PhaseImageDataset(
        image_dir=image_dir,
        annotations_file=ann_file,
        split=split_train,
        transform=train_transform,
    )
    val_ds = PhaseImageDataset(
        image_dir=image_dir,
        annotations_file=ann_file,
        split=split_val,
        transform=val_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        shuffle=False
    )
    return train_loader, val_loader

def setup_model_and_optimizer(ann_file, image_dir, split_train, num_classes, device, lr=1e-4):
    """Set up the model, loss function, and optimizer with layer4 and FC fine-tuning only."""
    model = PhaseResNet50Model(pretrained=True, num_classes=num_classes)
    
    # First, freeze ALL backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    # Then, unfreeze only layer4 parameters
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
    
    # Ensure the classifier (FC layer) is trainable (should be by default)
    for param in model.fc_phase.parameters():
        param.requires_grad = True
    
    model.to(device)
    
    # Print detailed information about trainable parameters
    trainable_params = []
    frozen_params = []
    total_trainable = 0
    total_frozen = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            total_trainable += param.numel()
        else:
            frozen_params.append(name)
            total_frozen += param.numel()
    
    print(f"\n=== MODEL PARAMETER STATUS ===")
    print(f"Total trainable parameters: {total_trainable:,}")
    print(f"Total frozen parameters: {total_frozen:,}")
    print(f"Percentage trainable: {100 * total_trainable / (total_trainable + total_frozen):.1f}%")
    
    print(f"\nTrainable layers:")
    for name in trainable_params:
        print(f"  - {name}")
    
    print(f"\nFrozen layers (first 10):")
    for name in frozen_params[:10]:
        print(f"  - {name}")
    if len(frozen_params) > 10:
        print(f"  ... and {len(frozen_params) - 10} more frozen layers")

    weights = compute_class_weights(
        annotations_file=ann_file,
        image_dir=image_dir,
        split=split_train,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    return model, criterion, optimizer

def plot_training_history(history, output_dir="plots"):
    """Plot and save training and validation loss and accuracy."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="Val Acc", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to {plot_path}")

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience, ckpt_path="bresnet50_best.pth"):
    """Run the training loop with early stopping and scheduler."""
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                print(f"  → New best model saved to {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  → No improvement in validation loss for {patience_counter} epochs")
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs!")
                break

    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.2f}%")
    return history

def main():
    """Main function to run the training pipeline."""
    IMAGE_DIR = "dataset/images"
    ANN_FILE = "dataset/annotations/image_annotations.csv"
    SPLIT_TRAIN = "train"
    SPLIT_VAL = "val"
    NUM_CLASSES = 35
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_EPOCHS = 2
    PATIENCE = 3
    CKPT_PATH = "backbone/resnet50_v2/models/resnet50_finetune.pth"

    device = setup_device()
    train_loader, val_loader = setup_datasets_and_loaders(
        IMAGE_DIR, ANN_FILE, SPLIT_TRAIN, SPLIT_VAL, BATCH_SIZE, NUM_WORKERS
    )
    model, criterion, optimizer = setup_model_and_optimizer(
        ANN_FILE, IMAGE_DIR, SPLIT_TRAIN, NUM_CLASSES, device
    )
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, NUM_EPOCHS, PATIENCE, CKPT_PATH
    )
    plot_training_history(history)

if __name__ == "__main__":
    main()