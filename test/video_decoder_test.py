import torch
from torchcodec.decoders import VideoDecoder

vc = VideoDecoder("dataset/videos/case_0994.mp4")

print(vc.metadata)

batch = []
# plot the first frame
for i in range(2000, 3000, 1):
    batch.append(vc[i])

# sample = batch_transform(torch.stack(batch))
torch_batch = torch.stack(batch)

print(torch_batch.shape)
# plt.imshow(torch_batch[900].permute(1, 2, 0).cpu().numpy() / 255)
# plt.show()