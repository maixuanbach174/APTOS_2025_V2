import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
# Assuming these imports are correct and available
from backbone.resnet50_v2.resnet50_finetune import PhaseResNet50Model
from phase_recognition.mstcn import MultiStageModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Cấu hình
config = {
    'num_stages': 2,
    'num_layers': 8,
    'num_f_maps': 32,
    'dim': 2048,
    'num_classes': 35,
    'causal_conv': True,
    'batch_size': 16,
    'seq_len': 256, # This should be the fixed length
    'image_size': 224
}

class EndToEndDataset(Dataset):
    def __init__(self, annotation_csv, image_root_dir, split="val", seq_len=config['seq_len'], transform=None): # Use config value
        """
        Args:
            annotation_csv: Path to CSV file with annotations
            image_root_dir: Directory with all the images
            split: 'train', 'val', or 'test'
            seq_len: Length of sequence
            transform: Optional transform to be applied on images
        """
        self.annotations = pd.read_csv(annotation_csv)
        self.annotations = self.annotations[self.annotations['split'] == split]
        self.image_root_dir = image_root_dir
        self.seq_len = seq_len
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Nhóm các frames theo video
        self.video_groups = self.annotations.groupby('video_id')
        self.video_ids = list(self.video_groups.groups.keys())

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_frames_df = self.video_groups.get_group(self.video_ids[idx]).sort_values('frame_idx')
        
        # Load all images and labels for the current video first
        loaded_images = []
        loaded_labels = []
        for _, row in video_frames_df.iterrows():
            image_path = os.path.join(self.image_root_dir, f'phase_{row["phase"]:02d}', row['image_name'])
            image = Image.open(image_path).convert('RGB')
            loaded_images.append(self.transform(image))
            loaded_labels.append(row['phase'])

        # Convert lists to tensors
        # Stack all loaded images first. This will have variable length T_original
        images_tensor_original = torch.stack(loaded_images) # Shape: [T_original, C, H, W]
        labels_tensor_original = torch.tensor(loaded_labels, dtype=torch.long) # Shape: [T_original]

        T_original = images_tensor_original.shape[0]

        # Apply random crop or pad to self.seq_len
        if T_original >= self.seq_len:
            # Randomly crop if the original sequence is longer
            start_idx = np.random.randint(0, T_original - self.seq_len + 1)
            images_final = images_tensor_original[start_idx : start_idx + self.seq_len]
            labels_final = labels_tensor_original[start_idx : start_idx + self.seq_len]
        else:
            # Pad at the end if the original sequence is shorter
            pad_length = self.seq_len - T_original
            
            # Pad images with zeros
            # torch.zeros takes arguments for the *shape* of the tensor to create
            # images_tensor_original.shape[1:] gives (C, H, W)
            pad_image_shape = (pad_length, *images_tensor_original.shape[1:])
            images_final = torch.cat([images_tensor_original, 
                                      torch.zeros(pad_image_shape, dtype=images_tensor_original.dtype)], 
                                     dim=0)
            
            # Pad labels with the last label or a special ignore index (e.g., -100 for CrossEntropyLoss)
            # Using the last label for simplicity, but consider a specific ignore_index if appropriate for your loss function.
            labels_final = torch.cat([labels_tensor_original, 
                                      torch.full((pad_length,), labels_tensor_original[-1].item(), dtype=labels_tensor_original.dtype)], 
                                     dim=0)
            
        return images_final, labels_final

class EndToEndModel(nn.Module):
    def __init__(self, feature_extractor, temporal_model):
        super(EndToEndModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.temporal_model = temporal_model

    def forward(self, x):
        # x shape: [batch_size, seq_len, c, h, w]
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape to process each frame independently by the feature extractor
        # New shape: [batch_size * seq_len, c, h, w]
        x_reshaped = x.view(-1, c, h, w) 
        
        # Extract features
        # features shape: [batch_size * seq_len, dim] (where dim is 2048)
        features = self.feature_extractor(x_reshaped)
        
        # Reshape back for the temporal model
        # New shape: [batch_size, dim, seq_len] - MSTCN expects (N, C_in, L) where C_in is features dim
        features_for_mstcn = features.view(batch_size, seq_len, -1).transpose(1, 2) # [batch_size, dim, seq_len]
        
        # MSTCN processing
        out = self.temporal_model(features_for_mstcn) # out shape: [num_stages, batch_size, num_classes, seq_len]
        return out

def plot_confusion_matrix(y_true, y_pred, num_classes):
    """Vẽ confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    # Ensure all possible classes are covered for ticks
    labels = [str(i) for i in range(num_classes)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Phase')
    plt.ylabel('True Phase')
    plt.title('Confusion Matrix')
    plt.savefig('visualizations/confusion_matrix_end_to_end.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_class_accuracies(report):
    """Vẽ accuracy của từng class"""
    class_f1_scores = {}
    for i in range(config['num_classes']):
        class_str = str(i)
        if class_str in report and 'f1-score' in report[class_str]:
            class_f1_scores[f'Phase {i}'] = report[class_str]['f1-score']
        else:
            class_f1_scores[f'Phase {i}'] = 0.0 # Handle cases where a class might not appear in report if no samples/predictions

    plt.figure(figsize=(15, 8))
    phases = list(class_f1_scores.keys())
    scores = list(class_f1_scores.values())
    
    plt.bar(phases, scores)
    plt.xticks(rotation=90, ha='right', fontsize=8) # Rotate labels for better readability
    plt.xlabel('Phase')
    plt.ylabel('F1-Score')
    plt.title('F1-Score for Each Phase')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/phase_accuracies_end_to_end.png', dpi=300)
    plt.close()

def evaluate_end_to_end():
    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    # Tạo thư mục lưu kết quả
    os.makedirs('visualizations', exist_ok=True)

    # Load ResNet model
    resnet = PhaseResNet50Model(pretrained=True)
    ckpt = torch.load(os.path.join("backbone/resnet50_v2/models", "resnet50_finetune.pth"), map_location=device)
    resnet.load_state_dict(ckpt)
    resnet.fc = nn.Identity() # Remove the classification head for feature extraction
    
    # Load MSTCN model
    mstcn = MultiStageModel(
        num_stages=config['num_stages'],
        num_layers=config['num_layers'],
        num_f_maps=config['num_f_maps'],
        dim=config['dim'],
        num_classes=config['num_classes'],
        causal_conv=config['causal_conv']
    )
    # Ensure correct path to MSTCN model
    mstcn_model_path = os.path.join('phase_recognition', 'models', 'mstcn_model.pth')
    if not os.path.exists(mstcn_model_path):
        print(f"Error: MSTCN model not found at {mstcn_model_path}. Please check the path.")
        return
        
    mstcn_checkpoint = torch.load(mstcn_model_path, map_location=device, weights_only=False)
    # Load state dict, ignoring non-matching keys if any (e.g., from DataParallel)
    mstcn.load_state_dict(mstcn_checkpoint['model_state_dict']) # Use the 'model_state_dict' key if present

    # Combine models
    model = EndToEndModel(resnet, mstcn).to(device)
    model.eval() # Set to evaluation mode

    # Load test dataset
    test_dataset = EndToEndDataset(
        annotation_csv="dataset/annotations/image_annotations.csv",
        image_root_dir="dataset/images/val",
        split="val",
        seq_len=config['seq_len'] # Pass the fixed sequence length
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=os.cpu_count() // 2 or 1 # Use half of CPU cores for data loading
    )

    # Evaluation
    all_predictions = []
    all_targets = []
    
    print("Bắt đầu đánh giá model end-to-end...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            predictions = model(images)  # [num_stages, batch_size, num_classes, seq_len]
            
            # Lấy predictions từ stage cuối cùng (predictions[-1])
            # Then permute to [batch_size, seq_len, num_classes] for argmax over classes
            # Then argmax over the class dimension (dim=2) to get [batch_size, seq_len]
            # No, predictions[-1] is already [batch_size, num_classes, seq_len]
            # So, argmax over dim=1 (num_classes)
            final_predictions = torch.argmax(predictions[-1], dim=1) # [batch_size, seq_len]
            
            # Flatten predictions and targets for sklearn metrics
            all_predictions.extend(final_predictions.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())
    
    # Tính toán và hiển thị metrics
    print("\nPhase Recognition Report:")
    # Generate labels for the classification report explicitly
    target_names = [f'Phase {i}' for i in range(config['num_classes'])]
    report = classification_report(all_targets, all_predictions, output_dict=True, target_names=target_names, zero_division=0)
    print(classification_report(all_targets, all_predictions, target_names=target_names, zero_division=0))
    
    # Lưu report vào file
    with open('visualizations/end_to_end_evaluation_report.txt', 'w') as f:
        f.write(classification_report(all_targets, all_predictions, target_names=target_names, zero_division=0))
    
    # Vẽ confusion matrix và accuracies
    plot_confusion_matrix(all_targets, all_predictions, config['num_classes'])
    plot_class_accuracies(report)
    
    print("\nĐã lưu kết quả đánh giá:")
    print("- Confusion Matrix: visualizations/confusion_matrix_end_to_end.png")
    print("- Phase Accuracies: visualizations/phase_accuracies_end_to_end.png")
    print("- Detailed Report: visualizations/end_to_end_evaluation_report.txt")

if __name__ == "__main__":
    evaluate_end_to_end()