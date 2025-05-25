from torchcodec.decoders import VideoDecoder
import matplotlib.pyplot as plt
from backbone.resnet50.aptos_dataset import get_resnet50_transform

vc = VideoDecoder("dataset/videos/case_0272.mp4")

print(vc.metadata)

# plot the first frame

sample = vc[2000]
transform = get_resnet50_transform()
sample = transform(sample)
print(sample.shape)
plt.imshow(sample.permute(1, 2, 0))
plt.show()