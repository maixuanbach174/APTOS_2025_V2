from backbone.resnet50_v2.dataset import PhaseImageDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import random

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id) 

def main():
    dataset = PhaseImageDataset(image_dir="dataset/images", annotations_file="dataset/annotations/image_annotations.csv", split="train")
    data_loader = DataLoader(dataset, batch_size=10, num_workers=4, worker_init_fn=worker_init_fn, shuffle=True)

    # Get one batch of data
    frame, label = next(iter(data_loader))  
    print(f"Batch shape: {frame.shape}")

    # Create a figure with 2x5 subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()

    # Plot each sample in the batch
    for i in range(10):
        axes[i].imshow(frame[i].permute(1, 2, 0))
        axes[i].set_title(f"Label: {label[i]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()