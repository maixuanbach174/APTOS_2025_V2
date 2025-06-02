import torch
from torchcodec.decoders import VideoDecoder
import matplotlib.pyplot as plt

vc = VideoDecoder("dataset/videos/case_1944.mp4")
frame = []
# for i in range(2000, 2010, 1):
#     frame.append(vc[i])

print(vc.metadata)
# timestamps = vc.metadata.seconds

# # Create a figure with 2x5 subplots
# fig, axes = plt.subplots(2, 5, figsize=(20, 8))
# axes = axes.ravel()

# # Plot each sample in the batch
# for i in range(10):
#     axes[i].imshow(frame[i].permute(1, 2, 0))
#     axes[i].axis('off')

# plt.tight_layout()
# plt.show()