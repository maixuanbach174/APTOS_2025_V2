import os
import glob
import csv
import random
from typing import Iterator, Tuple, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchcodec.decoders import VideoDecoder
import torchvision.transforms as T


def get_resnet50_transform() -> T.Compose:
    """Returns the standard ImageNet preprocessing pipeline for ResNet-50."""
    return T.Compose([
        T.ConvertImageDtype(torch.float),  # uint8 [0,255] -> float [0.0,1.0]
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


class AptosIterableDataset(IterableDataset):
    """
    IterableDataset that streams frames from videos, with optional
    shuffling of video order and/or per-video frame order.
    Supports multi-worker partitioning via internal logic in __iter__.
    Yields (frame_tensor, label, timestamp).
    """
    def __init__(
        self,
        video_dir: str,
        annotations_file: str,
        split: str = 'train',
        transform: Optional[T.Compose] = None,
        shuffle_videos: bool = False,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.transform = transform or get_resnet50_transform()
        self.split = split.lower()
        self.shuffle_videos = shuffle_videos

        # Load annotations filtered by split
        self.annotations = {}  # vid -> [(start, end, phase_id), ...]
        with open(annotations_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('split','train').lower() != self.split:
                    continue
                vid = row['video_id']
                start = float(row['start'])
                end   = float(row['end'])
                phase = int(row['phase_id'])
                self.annotations.setdefault(vid, []).append((start, end, phase))

        # Collect video file paths for this split
        all_videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        self.video_files = []
        for path in all_videos:
            vid = os.path.splitext(os.path.basename(path))[0]
            if vid in self.annotations:
                self.video_files.append(path)
        if not self.video_files:
            raise ValueError(f"No videos found for split='{self.split}' in {video_dir}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int, float]]:
        # Partition video list across workers
        worker_info = get_worker_info()
        if worker_info is None:
            vids = list(self.video_files)
        else:
            # Split video list evenly among workers
            vids = self.video_files[worker_info.id::worker_info.num_workers]

        # Optionally shuffle video order
        if self.shuffle_videos:
            random.shuffle(vids)

        # Stream frames from each assigned video
        for path in vids:
            vid = os.path.splitext(os.path.basename(path))[0]
            annots = self.annotations.get(vid, [])

            reader = VideoDecoder(path, seek_mode='exact')
            num_frames = reader.metadata.num_frames

            for idx in range(num_frames):
                try:
                    frame_obj = reader.get_frame_at(idx)
                    frame = frame_obj.data
                    timestamp = float(frame_obj.pts_seconds)
                except RuntimeError:
                    break
                # Lookup label
                label = None
                for start, end, phase in annots:
                    if start <= timestamp < end:
                        label = phase
                        break
                if label is None:
                    continue
                # Apply transform
                frame = self.transform(frame)

                yield frame, label, timestamp
