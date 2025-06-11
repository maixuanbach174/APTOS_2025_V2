import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class OphNetFeatureDataset(Dataset):
    """Dataset for MS-TCN training on ResNet50-extracted features.
    
    Expects your CSV to have columns:
      - video_id, frame_idx, phase, augment, feature_path
    and that for each video_id there’s exactly one .npy file
    with shape [2048, T] saved to feature_path.
    
    Args:
        annotation_csv (str): Path to CSV with image_annotations_with_features.csv
        seq_len (int, optional): If set, each sample is a random crop
            (or zero‐pad) of length seq_len along the time axis.
        transform (callable, optional): Receives (feats, labels) and
            returns transformed versions.
    """
    def __init__(self, annotation_csv, seq_len=None, split=None, transform=None):
        self.df = pd.read_csv(annotation_csv)
        self.seq_len = seq_len
        self.split = split
        self.transform = transform
        # filter out augmented images
        self.df = self.df[self.df['feature_path'].notna()]
        if split is not None:
            self.df = self.df[self.df['split'] == split]
        # build list of (video_id, feature_path, labels_array)
        self.videos = []
        for vid, g in self.df.groupby("video_id"):
            g = g.sort_values("frame_idx")
            # There should be exactly one unique .npy per video:
            feat_paths = g["feature_path"].unique()
            assert len(feat_paths) == 1, \
                f"Expected one .npy per video, got {len(feat_paths)} for {vid}"
            feat_path = feat_paths[0]

            labels = g["phase"].values.astype(np.int64)  # shape [T,]
            self.videos.append((vid, feat_path, labels))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid, feat_path, labels = self.videos[idx]

        # load once per video: feats shape = [2048, T]
        feats = np.load(feat_path)
        # ensure dims
        assert feats.ndim == 2 and feats.shape[0] == 2048, \
            f"Expected [2048, T], got {feats.shape} for {vid}"
        T = feats.shape[1]

        # random crop or pad to seq_len
        if self.seq_len is not None:
            L = self.seq_len
            if T >= L:
                start = np.random.randint(0, T - L + 1)
                feats = feats[:, start : start + L]
                labels_crop = labels[start : start + L]
            else:
                # pad at end with zeros / last label
                pad = L - T
                feats = np.concatenate(
                    [feats, np.zeros((2048, pad), dtype=feats.dtype)],
                    axis=1
                )
                labels_crop = np.concatenate(
                    [labels, np.full((pad,), labels[-1], dtype=labels.dtype)],
                    axis=0
                )
            labels = labels_crop

        # to torch
        feats = torch.from_numpy(feats).float()    # [2048, seq_len or T]
        labels = torch.from_numpy(labels).long()   # [seq_len or T]

        if self.transform:
            feats, labels = self.transform(feats, labels)

        return feats, labels
    
if __name__ == "__main__":
    # Initialize the dataset with a fixed sequence length
    # For example, let's say 256 frames is a suitable length
    dataset = OphNetFeatureDataset(
        annotation_csv="dataset/annotations/image_annotations_with_features.csv",
        split="val",
        seq_len=256 # <--- IMPORTANT: Set a fixed sequence length
    )
    frame_count = 0
    print(len(dataset))
    for i in range(len(dataset)):
        feats, labels = dataset[i]
        frame_count += feats.shape[1]
    print(frame_count)