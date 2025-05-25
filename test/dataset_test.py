from backbone.resnet50.aptos_dataset import AptosDataset
import time
import psutil
import os
from torch.utils.data import DataLoader


def get_memory_usage():
    """Get the memory usage of the current process in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

dataset = AptosDataset("dataset/videos", "dataset/annotations/APTOS_train-val_annotation.csv", batch_size=16)
loader = DataLoader(dataset, batch_size=None)

start_time = time.time()
start_mem = get_memory_usage()
print(f"Initial memory usage: {start_mem:.2f} MB")

frame_count = 0
for frame, label, timestamp in loader:
    if frame_count % 100 == 0:  # Print every 100 frames
        current_mem = get_memory_usage()
        print(f"Frame {frame_count}:")
        print(f"Current memory usage: {current_mem:.2f} MB")
        print(f"Memory increase: {(current_mem - start_mem):.2f} MB")
        print("-" * 50)
    frame_count += 1

end_time = time.time()
end_mem = get_memory_usage()

print(f"\nFinal Statistics:")
print(f"Total frames processed: {frame_count}")
print(f"Time taken: {end_time - start_time:.2f} seconds")
print(f"Frames per second: {frame_count / (end_time - start_time):.2f}")
print(f"Final memory usage: {end_mem:.2f} MB")
print(f"Total memory increase: {(end_mem - start_mem):.2f} MB")
