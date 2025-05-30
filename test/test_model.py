from backbone.resnet50.resnet50_finetune import PhaseResNet50Model
from backbone.resnet50.aptos_dataset import AptosIterableDataset
from torch.utils.data import DataLoader
from backbone.resnet50.utils import compute_duration_weights
import torch.nn as nn
if __name__ == "__main__":
    annotations_file = 'dataset/annotations/APTOS_train-val_annotation.csv'
    video_dir = 'dataset/videos'
    split = 'train'
    num_classes = 35
    batch_size = 16
    num_workers = 4

    dataset = AptosIterableDataset(
        annotations_file=annotations_file,
        video_dir=video_dir,
        split=split
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


    frame, label, timestamp = next(iter(dataloader))

    print(frame.shape)
    print(label.shape)
    print(timestamp.shape)

    model = PhaseResNet50Model(pretrained=True, num_classes=num_classes)
    model.eval()

    features, logits = model(frame)
    loss_weights = compute_duration_weights(
        annotations_file=annotations_file,
        video_dir=video_dir,
        split=split,
        num_classes=num_classes
    )

    loss = nn.CrossEntropyLoss(weight=loss_weights)
    loss_value = loss(logits, label)

    print(loss_value)