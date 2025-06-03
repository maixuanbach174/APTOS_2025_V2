import os
import csv
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchcodec.decoders import VideoDecoder

# Configuration
CSV_FILE = "dataset/annotations/samples_mapping.csv"
OUTPUT_IMAGE_DIR = "dataset/train_images"
VIDEO_DIR = "dataset/videos"

# Augmentation pipeline for augmented frames
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomRotation(15),
])

# Create output directory
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# Group samples by video_id to minimize video loading/unloading
video_samples = {}
with open(CSV_FILE, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_id = row['video_id']
        video_samples.setdefault(video_id, []).append({
            'frame_idx': int(row['frame_idx']),
            'phase': int(row['phase']),
            'augment': row['augment'].lower() == 'true'
        })

# Process each video
total_processed = 0
for video_id, samples in video_samples.items():
    print(f"Processing video {video_id} ({len(samples)} frames)...")
    
    # Open video
    video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        print(f"Warning: Video file {video_path} not found, skipping...")
        continue
    
    try:
        reader = VideoDecoder(video_path)
        
        # Process each frame
        for sample in samples:
            # Create phase directory if needed
            phase_dir = os.path.join(OUTPUT_IMAGE_DIR, f"phase_{sample['phase']:02d}")
            os.makedirs(phase_dir, exist_ok=True)
            
            # Extract frame
            frame_obj = reader.get_frame_at(sample['frame_idx'])
            frame = frame_obj.data  # Torch tensor (C,H,W)
            
            # Convert to PIL for saving/augmentation
            pil_img = to_pil_image(frame)
            
            # Apply augmentation if needed
            if sample['augment']:
                pil_img = augment_transform(pil_img)
            
            # Save image
            img_name = f"{video_id}_{sample['frame_idx']:06d}{'_aug' if sample['augment'] else ''}.png"
            save_path = os.path.join(phase_dir, img_name)
            pil_img.save(save_path)
            
            total_processed += 1
            if total_processed % 100 == 0:
                print(f"Processed {total_processed} frames...")
        
    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
        continue

print(f"Done! Successfully processed {total_processed} frames.")
