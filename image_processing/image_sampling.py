#______________________________________________________________________________

import os
import csv
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchcodec.decoders import VideoDecoder
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configuration
TRAIN_CSV_FILE = "dataset/annotations/train_samples_mapping.csv"
VAL_CSV_FILE = "dataset/annotations/val_samples_mapping.csv"
OUTPUT_IMAGE_DIR = "dataset/images"  # Changed from train_images since it contains both splits
VIDEO_DIR = "dataset/videos"
IMAGE_ANNOTATION_FILE = "dataset/annotations/image_annotations.csv"
# Annotation file contains the following columns:
# video_id, frame_idx, image_name, phase_id, augment, timestamp

# Augmentation pipeline for augmented frames
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomRotation(15),
])

# Create output directories
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, "val"), exist_ok=True)

def save_image_worker(args):
    img, save_path = args
    img.save(save_path)

def process_one_video(args):
    video_id, samples = args
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        return [], 0
    print(f"Processing video {video_id} with {len(samples)} samples")
    reader = VideoDecoder(video_path, seek_mode="approximate")
    # Sort samples by frame_idx to avoid FFmpeg's random seeks
    samples_sorted = sorted(samples, key=lambda s: s["frame_idx"])
    idx_list = [s["frame_idx"] for s in samples_sorted]

    # Batch‐decode once
    frame_objs = reader.get_frames_at(indices=idx_list)
    processed = 0
    annotations = []
    save_tasks = []
    
    # Create phase directories for each split first
    phase_dirs = {(s["split"], s["phase"]) for s in samples_sorted}
    for split, phase in phase_dirs:
        os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, split, f"phase_{phase:02d}"), exist_ok=True)
    
    # Process frames
    for s, frame_obj in zip(samples_sorted, frame_objs):
        phase = s["phase"]
        is_aug = s["augment"]
        split = s["split"]
        img = to_pil_image(frame_obj.data)
        if is_aug:
            img = augment_transform(img)

        img_name = f"{video_id}_{s['frame_idx']:06d}{'_aug' if is_aug else ''}.png"
        save_path = os.path.join(OUTPUT_IMAGE_DIR, split, f"phase_{phase:02d}", img_name)
        save_tasks.append((img, save_path))
        
        annotations.append({
            "video_id": video_id,
            "frame_idx": s["frame_idx"],
            "image_name": img_name,
            "split": split,
            "phase": phase,
            "augment": is_aug,
            "timestamp": round(float(frame_obj.pts_seconds), 2)
        })
        processed += 1
    
    # Save images in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(save_image_worker, save_tasks))
    
    return annotations, processed

def write_annotations_batch(annotations, filename, mode='w'):
    with open(filename, mode, newline="", buffering=8192) as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id", "frame_idx", "image_name", "split", "phase", "augment", "timestamp"
        ])
        if mode == 'w':
            writer.writeheader()
        writer.writerows(annotations)

if __name__ == "__main__":
    # 1) Read CSV into video_samples dict
    print("Reading sample mapping files...")
    video_samples = {}
    
    # Read training samples
    print("Reading training samples...")
    with open(TRAIN_CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            video_samples.setdefault(vid, []).append({
                "frame_idx": int(row["frame_idx"]),
                "phase": int(row["phase"]),
                "augment": row["augment"].lower() == "true",
                "split": "train"
            })
    
    # Read validation samples
    print("Reading validation samples...")
    with open(VAL_CSV_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            video_samples.setdefault(vid, []).append({
                "frame_idx": int(row["frame_idx"]),
                "phase": int(row["phase"]),
                "augment": row["augment"].lower() == "true",
                "split": "val"
            })

    # 2) Build a list of (video_id, samples) tuples
    tasks = list(video_samples.items())
    print(f"Found {len(tasks)} videos to process")
    
    # 3) Process videos in parallel with progress bar
    n_processes = max(1, int(cpu_count() * 0.75))  # Use 75% of CPUs
    print(f"Using {n_processes} processes")
    
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_one_video, tasks),
            total=len(tasks),
            desc="Processing videos"
        ))
    
    # 4) Collect and save results
    print("Saving annotations...")
    all_annotations = []
    total_processed = 0
    for annotations, processed in results:
        all_annotations.extend(annotations)
        total_processed += processed
    
    write_annotations_batch(all_annotations, IMAGE_ANNOTATION_FILE)
    
    # Print summary
    train_count = sum(1 for ann in all_annotations if ann["split"] == "train")
    val_count = sum(1 for ann in all_annotations if ann["split"] == "val")
    print(f"\nProcessing complete!")
    print(f"Total frames processed: {total_processed}")
    print(f"Training frames: {train_count}")
    print(f"Validation frames: {val_count}")
    print(f"Image annotations saved to {IMAGE_ANNOTATION_FILE}")
