from backbone.resnet50.aptos_dataset import AptosIterableDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

dataset = AptosIterableDataset("dataset/videos", "dataset/annotations/APTOS_train-val_annotation.csv", split="train",shuffle_frames=True, shuffle_videos=True)
data_loader = DataLoader(dataset, batch_size=10)

# Get one batch of data
frame, label, timestamp = next(iter(data_loader))
print(f"Batch shape: {frame.shape}")

# Create a figure with 2x5 subplots
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

# Plot each sample in the batch
for i in range(10):
    axes[i].imshow(frame[i].permute(1, 2, 0))
    axes[i].set_title(f"Label: {label[i]}\nTimestamp: {timestamp[i]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
