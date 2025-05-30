from backbone.resnet50.aptos_dataset import AptosIterableDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import random

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id) 

def main():
    dataset = AptosIterableDataset("dataset/videos", "dataset/annotations/APTOS_train-val_annotation.csv", split="train", shuffle_videos=True)
    data_loader = DataLoader(dataset, batch_size=10, num_workers=4, worker_init_fn=worker_init_fn)

    skip_batch = 100

    data_loader = iter(data_loader)

    for i in range(skip_batch):
        print(i)
        next(data_loader)

    # Get one batch of data
    frame, label, timestamp = next(data_loader)
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

if __name__ == "__main__":
    main()