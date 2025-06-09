import os
import csv
from PIL import Image
import torch
import torchvision.transforms as T


def get_resnet50_transform() -> T.Compose:
    """Returns the standard ImageNet preprocessing pipeline for ResNet-50."""
    return T.Compose([
        T.ToTensor(),
        T.ConvertImageDtype(torch.float),  # uint8 [0,255] -> float [0.0,1.0]
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

class PhaseImageDataset:
    """
    A dataset class for lazy-loading phase images based on an annotation CSV file and a directory structure:
      image_dir/<split>/phase_<XX>/<image_name>
    Each sample returned is a tuple (image, phase_label).
    """

    def __init__(self, annotations_file: str, image_dir: str, split: str = "train", transform=None):
        """
        annotations_file: Path to the CSV file containing at least 'image_name', 'phase', and optional 'split' columns.
        image_dir: Root directory containing subdirectories for each split, e.g. "dataset/images".
        split: Which split to load ('train', 'val', etc.). This should match values in the 'split' column of the CSV.
        transform: Optional callable to apply transformations to PIL images before returning.
        """
        self.annotations_file = annotations_file
        self.image_dir = image_dir
        self.split = split.strip().lower()
        if transform is None:
            self.transform = get_resnet50_transform()
        else:
            self.transform = transform

        self.samples = []
        with open(self.annotations_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                row_split = row.get("split", "train").strip().lower()
                if row_split != self.split:
                    continue

                phase_id = int(row["phase"])
                image_name = row.get("image_name", "").strip()

                phase_str = f"{phase_id:02d}"
                folder_path = os.path.join(self.image_dir, self.split, f"phase_{phase_str}")
                img_path = os.path.join(folder_path, image_name)

                if os.path.isfile(img_path):
                    self.samples.append({
                        "img_path": img_path,
                        "phase_id": phase_id
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Lazily load the image at the given index. Returns (image, phase_label).
        `image` is a PIL.Image.Image instance (after optional transform).
        `phase_label` is an integer phase ID.
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError("Index out of range.")

        sample = self.samples[idx]
        img_path = sample["img_path"]
        phase_id = sample["phase_id"]

        # Lazy-load the image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, phase_id
