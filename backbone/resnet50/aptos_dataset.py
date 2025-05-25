import csv
import glob
import os
from typing import Iterator, Tuple

import torch
from torch.utils.data import IterableDataset
from torchcodec.decoders import VideoDecoder
import torchvision.transforms as T

def get_resnet50_transform() -> T.Compose:
    """Returns the standard ImageNet preprocessing pipeline for ResNet-50."""
    return T.Compose([
        T.ConvertImageDtype(torch.float),  # convert uint8 [0,255] to float [0.0,1.0]
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class AptosDataset(IterableDataset):
    """
    An iterable dataset that yields batches of frames directly from videos.
    Uses torchcodec's VideoDecoder for efficient batch decoding.
    """
    def __init__(self, video_dir: str, annotations_file: str, transform=get_resnet50_transform(), batch_size: int = 32):
        """
        Args:
            video_dir (str): Directory containing video files
            annotations_file (str): Path to annotations CSV file
            transform (callable, optional): Optional transform to be applied on frames
            batch_size (int): Size of frame batches to return
        """
        self.video_dir = video_dir
        self.transform = transform
        self.batch_size = batch_size
        
        # Collect all mp4 files in directory
        self.video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        if not self.video_files:
            raise ValueError(f"No .mp4 files found in {video_dir}")
        
        self.annotations = {}  # video_id -> list of (start, end, phase_id)
        with open(annotations_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                vid = row['video_id']
                start = float(row['start'])
                end = float(row['end'])
                phase = row['phase_id']
                self.annotations.setdefault(vid, []).append((start, end, phase))
        
        self.video_id = self.video_files[0].split("/")[-1].split(".")[0]
        
        # Initialize pointers
        self.current_video_idx = 0
        self.current_reader = VideoDecoder(self.video_files[0])
        self.current_frame_idx = 0

    def _load_next_video(self) -> None:
        """Load the next video in the sequence."""
        self.current_video_idx = (self.current_video_idx + 1) % len(self.video_files)
        next_path = self.video_files[self.current_video_idx]
        self.video_id = next_path.split("/")[-1].split(".")[0]
        self.current_reader = VideoDecoder(next_path)
        self.current_frame_idx = 0

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Returns an iterator that yields batches of (frames, labels, timestamps)."""
        self.current_video_idx = 0
        self.current_reader = VideoDecoder(self.video_files[0])
        self.current_frame_idx = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a batch of (frames, labels, timestamps)."""
        frames = []
        labels = []
        timestamps = []
        
        while len(frames) < self.batch_size:
            # Check if we need to move to next video
            if self.current_frame_idx >= self.current_reader.metadata.num_frames:
                if self.current_video_idx >= len(self.video_files) - 1:
                    if not frames:  # If we haven't collected any frames, stop iteration
                        raise StopIteration
                    break  # If we have some frames, return them as a partial batch
                self._load_next_video()
            
            # Get a batch of frames from current position
            remaining = self.batch_size - len(frames)
            end_idx = min(self.current_frame_idx + remaining, self.current_reader.metadata.num_frames)
            indices = list(range(self.current_frame_idx, end_idx))
            
            # Decode batch of frames
            frame_objs = self.current_reader.get_frames_at(indices=indices)
            
            # Process each frame in the batch
            for frame_obj in frame_objs:
                frame = frame_obj.data
                timestamp = float(frame_obj.pts_seconds)
                
                # Find label for this timestamp
                label = None
                for start, end, phase in self.annotations.get(self.video_id, []):
                    if start <= timestamp < end:
                        label = phase
                        break
                
                if label is not None:  # Only add frames that have labels
                    if self.transform:
                        frame = self.transform(frame)
                    frames.append(frame)
                    labels.append(int(label))
                    timestamps.append(timestamp)
            
            self.current_frame_idx = end_idx
        
        # Stack the collected frames into a batch
        frames_batch = torch.stack(frames)
        labels_batch = torch.tensor(labels, dtype=torch.long)
        # timestamps_batch = torch.tensor(timestamps, dtype=torch.float)

        # return frames_batch, labels_batch, timestamps_batch
        return frames_batch, labels_batch, timestamps
