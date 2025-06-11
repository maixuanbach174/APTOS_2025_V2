import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from phase_recognition.dataset import OphNetFeatureDataset
from phase_recognition.mstcn import MultiStageModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# Cấu hình model
config = {
    'num_stages': 2,
    'num_layers': 8,
    'num_f_maps': 32,
    'dim': 2048,
    'num_classes': 35,
    'causal_conv': True,
    'batch_size': 16,
    'seq_len': 256
}

def plot_confusion_matrix(y_true, y_pred, num_classes):
    """Vẽ confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Phase')
    plt.ylabel('True Phase')
    plt.title('Confusion Matrix')
    plt.savefig('visualizations/confusion_matrix_eval.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_class_accuracies(report):
    """Vẽ accuracy của từng class"""
    # Lấy metrics cho từng class
    class_metrics = {}
    for i in range(config['num_classes']):
        if str(i) in report:
            class_metrics[f'Phase {i}'] = report[str(i)]['f1-score']
    
    # Vẽ biểu đồ
    plt.figure(figsize=(15, 8))
    phases = list(class_metrics.keys())
    scores = list(class_metrics.values())
    
    plt.bar(phases, scores)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Phase')
    plt.ylabel('F1-Score')
    plt.title('F1-Score for Each Phase')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('visualizations/phase_accuracies_eval.png', dpi=300)
    plt.close()

def evaluate_model():
    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Sử dụng device: {device}")

    # Tạo thư mục lưu kết quả
    os.makedirs('visualizations', exist_ok=True)

    # Load validation dataset
    val_dataset = OphNetFeatureDataset(
        annotation_csv="dataset/annotations/image_annotations_with_features.csv",
        split="val",
        seq_len=config['seq_len']
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

    # Load model weights
    checkpoint = torch.load('phase_recognition/models/mstcn_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")

    # Evaluation
    model.eval()
    all_predictions = []
    all_targets = []
    
    print("Bắt đầu đánh giá model...")
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            predictions = model(features)  # [num_stages, batch_size, num_classes, seq_len]
            
            # Lấy predictions từ stage cuối cùng
            final_predictions = torch.argmax(predictions[-1], dim=1)
            
            # Lưu predictions và targets
            all_predictions.extend(final_predictions.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())
    
    # Tính toán và hiển thị metrics
    print("\nPhase Recognition Report:")
    report = classification_report(all_targets, all_predictions, output_dict=True)
    print(classification_report(all_targets, all_predictions))
    
    # Lưu report vào file
    with open('visualizations/evaluation_report.txt', 'w') as f:
        f.write(classification_report(all_targets, all_predictions))
    
    # Vẽ confusion matrix
    plot_confusion_matrix(all_targets, all_predictions, config['num_classes'])
    
    # Vẽ accuracy của từng class
    plot_class_accuracies(report)
    
    print("\nĐã lưu kết quả đánh giá:")
    print("- Confusion Matrix: visualizations/confusion_matrix_eval.png")
    print("- Phase Accuracies: visualizations/phase_accuracies_eval.png")
    print("- Detailed Report: visualizations/evaluation_report.txt")

if __name__ == "__main__":
    evaluate_model()
