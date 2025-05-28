import os
import glob
import csv
from typing import Iterator, Tuple

import torch
from torch.utils.data import IterableDataset
from torchcodec.decoders import VideoDecoder
import torchvision.transforms as T


def get_resnet50_transform() -> T.Compose:
    """Returns the standard ImageNet preprocessing pipeline for ResNet-50."""
    return T.Compose([
        T.ConvertImageDtype(torch.float),  # uint8 [0,255] → float [0,1]
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class AptosDataset(IterableDataset):
    """
    Iterable dataset yielding batches of (frames, labels, timestamps) from videos,
    filtered by train/val split in the CSV annotation file.
    """
    def __init__(
        self,
        video_dir: str,
        annotations_file: str,
        split: str = 'train',
        transform: T.Compose = get_resnet50_transform(),
        batch_size: int = 32,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.transform = transform
        self.batch_size = batch_size
        self.split = split.lower()

        # Read annotations, filtering by split
        self.annotations = {}  # video_id -> list of (start, end, phase_id)
        with open(annotations_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('split', '').lower() != self.split:
                    continue
                vid = row['video_id']
                start = float(row['start'])
                end = float(row['end'])
                phase = row['phase_id']
                self.annotations.setdefault(vid, []).append((start, end, phase))

        # Gather video files that have annotations in this split
        all_mp4 = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
        self.video_files = []
        for path in all_mp4:
            vid = os.path.splitext(os.path.basename(path))[0]
            if vid in self.annotations:
                self.video_files.append(path)
        if not self.video_files:
            raise ValueError(f"No .mp4 files found for split='{self.split}' in {video_dir}")

        # Initialize first video reader
        self.current_video_idx = 0
        self.current_reader = VideoDecoder(self.video_files[0])
        self.current_frame_idx = 0

    def _load_next_video(self) -> None:
        """Advance to the next video in the split and reset frame index."""
        self.current_video_idx += 1
        if self.current_video_idx >= len(self.video_files):
            raise StopIteration
        path = self.video_files[self.current_video_idx]
        self.current_reader = VideoDecoder(path)
        self.current_frame_idx = 0

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Reset for new epoch
        self.current_video_idx = 0
        self.current_reader = VideoDecoder(self.video_files[0])
        self.current_frame_idx = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a batch of (frames, labels, timestamps)."""
        frames, labels, timestamps = [], [], []
        while len(frames) < self.batch_size:
            # If current video exhausted, move on
            if self.current_frame_idx >= self.current_reader.metadata.num_frames:
                self._load_next_video()

            # Determine indices for this batch chunk
            remaining = self.batch_size - len(frames)
            end_idx = min(self.current_frame_idx + remaining, self.current_reader.metadata.num_frames)
            indices = list(range(self.current_frame_idx, end_idx))

            # Decode frames in one go
            frame_objs = self.current_reader.get_frames_at(indices)

            for obj in frame_objs:
                frame = obj.data
                ts = float(obj.pts_seconds)
                # Lookup label
                label = None
                vid = os.path.splitext(os.path.basename(self.video_files[self.current_video_idx]))[0]
                for s, e, p in self.annotations.get(vid, []):
                    if s <= ts < e:
                        label = p
                        break
                if label is not None:
                    if self.transform:
                        frame = self.transform(frame)
                    frames.append(frame)
                    labels.append(int(label))
                    timestamps.append(ts)
            self.current_frame_idx = end_idx

        # Stack and return batch
        frames_batch = torch.stack(frames)
        labels_batch = torch.tensor(labels, dtype=torch.long)
        timestamps_batch = torch.tensor(timestamps)
        return frames_batch, labels_batch, timestamps_batch
