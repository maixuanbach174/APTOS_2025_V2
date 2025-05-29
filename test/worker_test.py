from backbone.resnet50.aptos_dataset import AptosIterableDataset
import time
import psutil
import os
from torch.utils.data import DataLoader
import torch
import random

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)

def get_memory_usage():
    """Get the memory usage of the current process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def main():
    dataset = AptosIterableDataset("dataset/videos", "dataset/annotations/APTOS_train-val_annotation.csv", split="train", shuffle_videos=False)
    loader = DataLoader(dataset, batch_size=16, num_workers=4, worker_init_fn=worker_init_fn)

    start_time = time.time()
    start_mem = get_memory_usage()
    print(f"Initial memory usage: {start_mem:.2f} MB")

    batch_count = 0
    for frame, label, timestamp in loader:
        if batch_count % 100 == 0:  # Print every 100 frames
            current_mem = get_memory_usage()
            print(f"Batch {batch_count}:")
            print(f"Current memory usage: {current_mem:.2f} MB")
            print(f"Memory increase: {(current_mem - start_mem):.2f} MB")
            print("-" * 50)
        batch_count += 1

    end_time = time.time()
    end_mem = get_memory_usage()

    print(f"\nFinal Statistics:")
    print(f"Total batches processed: {batch_count}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Batches per second: {batch_count / (end_time - start_time):.2f}")
    print(f"Final memory usage: {end_mem:.2f} MB")
    print(f"Total memory increase: {(end_mem - start_mem):.2f} MB")


if __name__ == "__main__":
    main()