import os
import csv
import numpy as np
import torch

def compute_class_weights(annotations_file: str, image_dir: str, split: str = 'train', num_classes: int = 35):
    num_classes = 35  
    image_counts = [0] * num_classes
    split_lower = split.strip().lower()

    phase_to_filenames = {}
    for phase_id in range(num_classes):
        phase_str = f"{phase_id:02d}"
        folder_path = os.path.join(image_dir, split_lower, f"phase_{phase_str}")

        if os.path.isdir(folder_path):
            phase_to_filenames[phase_id] = set(os.listdir(folder_path))
        else:
            phase_to_filenames[phase_id] = set()

    with open(annotations_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("split", "train").strip().lower() != split_lower:
                continue

            try:
                phase_id = int(row["phase"])
            except (KeyError, ValueError):
                continue
            if not (0 <= phase_id < num_classes):
                continue

            image_name = row.get("image_name", "").strip()
            if not image_name:
                continue

            if image_name in phase_to_filenames[phase_id]:
                image_counts[phase_id] += 1
    total_samples = sum(image_counts)
    weights = np.zeros(num_classes)
    epsilon = 1e-6
    for phase_id in range(num_classes):
        if image_counts[phase_id] > 0:
            weight = total_samples / (image_counts[phase_id] + epsilon) 
            weights[phase_id] = weight
        else:
            weights[phase_id] = 0.0
    non_zero_classes = np.sum(weights > 0)
    if weights.sum() > 0:
        weights = weights / np.sum(weights) * non_zero_classes
    return torch.tensor(weights, dtype=torch.float32)

if __name__ == "__main__":
    annotations_file = "dataset/annotations/image_annotations.csv"
    image_dir = "dataset/images"
    split = "train"
    num_classes = 35
    weights = compute_class_weights(annotations_file, image_dir, split, num_classes)
    print(weights)
