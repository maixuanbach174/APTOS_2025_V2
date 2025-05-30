import csv
import torch
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_phase_distribution(annotations_file: str, video_dir: str, split: str = 'train'):
    """
    Plot the distribution of surgical phases in the dataset.
    
    Args:
        annotations_file: CSV with columns [video_id, start, end, split, phase_id]
        video_dir: Directory containing the video files
        split: which split to use ('train' or 'val')
    """
    # Get valid video IDs
    # mp4s = glob.glob(os.path.join(video_dir, '*.mp4'))
    # valid_ids = {os.path.splitext(os.path.basename(p))[0] for p in mp4s}
    
    # Count clips per phase
    num_classes = 35  # Total number of surgical phases
    clip_counts = [0] * num_classes
    total_duration = [0.0] * num_classes  # Track duration for each phase
    
    with open(annotations_file, newline='') as f:
        rd = csv.DictReader(f)
        for row in rd:
            # if row['video_id'] not in valid_ids:
            #     continue
            if row.get('split','train').lower() != split.lower():
                continue
            phase = int(row['phase_id'])
            clip_counts[phase] += 1
            # Calculate duration of this clip
            duration = float(row['end']) - float(row['start'])
            total_duration[phase] += duration
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot clip counts
    phases = range(num_classes)
    ax1.bar(phases, clip_counts)
    ax1.set_title(f'Number of Clips per Phase ({split} split)')
    ax1.set_xlabel('Phase ID')
    ax1.set_ylabel('Number of Clips')
    # Set x-axis ticks for all phase IDs
    ax1.set_xticks(phases)
    ax1.set_xticklabels([str(i) for i in phases])
    
    # Add value labels on top of bars
    for i, count in enumerate(clip_counts):
        if count > 0:  # Only show labels for non-zero counts
            ax1.text(i, count, str(count), ha='center', va='bottom')
    
    # Plot total duration
    ax2.bar(phases, total_duration)
    ax2.set_title(f'Total Duration per Phase ({split} split)')
    ax2.set_xlabel('Phase ID')
    ax2.set_ylabel('Duration (seconds)')
    # Set x-axis ticks for all phase IDs
    ax2.set_xticks(phases)
    ax2.set_xticklabels([str(i) for i in phases])
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on top of bars (rounded to 1 decimal)
    for i, duration in enumerate(total_duration):
        if duration > 0:  # Only show labels for non-zero durations
            ax2.text(i, duration, f'{duration:.1f}s', ha='center', va='bottom')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'phase_distribution_{split}.png')
    plt.close()
    
    # Print summary statistics
    total_clips = sum(clip_counts)
    total_time = sum(total_duration)
    print(f"\nSummary for {split} split:")
    print(f"Total number of clips: {total_clips}")
    print(f"Total duration: {total_time:.1f} seconds ({total_time/3600:.1f} hours)")
    print(f"Number of phases present: {sum(1 for c in clip_counts if c > 0)}")


def compute_duration_weights(
    annotations_file: str,
    video_dir: str,
    split: str = 'train',
    num_classes: int = 35,
) -> torch.Tensor:
    """
    Compute per-class weights based on total annotated duration of each phase.
    Uses median-frequency balancing:
       w[c] = median_duration / duration_c   if duration_c > 0
            = 0                              otherwise
    then normalized so sum(w) = num_classes.
    """
    # 1) Gather valid video IDs
    # mp4s = glob.glob(os.path.join(video_dir, '*.mp4'))
    # valid_ids = { os.path.splitext(os.path.basename(p))[0] for p in mp4s }

    # 2) Accumulate total duration per phase
    duration_sums = [0.0] * num_classes
    with open(annotations_file, newline='') as f:
        rd = csv.DictReader(f)
        for row in rd:
            vid = row['video_id']
            # if vid not in valid_ids:
            #     continue
            if row.get('split','train').lower() != split.lower():
                continue
            if row['phase_id'] == '16':
                print(row)
            phase = int(row['phase_id'])
            start = float(row['start'])
            end   = float(row['end'])
            duration_sums[phase] += max(0.0, end - start)

    durations = torch.tensor(duration_sums, dtype=torch.float)

    # 3) Median-frequency balancing
    print(durations)
    nonzero = durations > 0
    if not nonzero.any():
        raise ValueError(f"No phases with duration > 0 in split='{split}'")
    median_dur = float(durations[nonzero].median())
    
    raw = torch.zeros_like(durations)
    raw[nonzero] = median_dur / durations[nonzero]
    # absent classes remain zero

    # 4) Normalize so sum(raw) == num_classes
    total = raw.sum().item()
    if total > 0:
        raw = raw * (num_classes / total)

    return raw

# Usage:
if __name__ == "__main__":
    # weights = compute_duration_weights(
    #     annotations_file='dataset/annotations/APTOS_train-val_annotation.csv',
    #     video_dir='dataset/videos',
    #     split='val',
    #     num_classes=35
    # )
    # print("Duration-based phase weights:", weights)

    # plot_phase_distribution(
    #     annotations_file='dataset/annotations/APTOS_train-val_annotation.csv',
    #     video_dir='dataset/videos',
    #     split='train'
    # )

    vid_tables = {}

    annotations_file = 'dataset/annotations/APTOS_train-val_annotation.csv'
    with open(annotations_file, newline='') as f:
        rd = csv.DictReader(f)
        for row in rd:
            vid = row['video_id']
            if vid not in vid_tables:
                vid_tables[vid] = 0
            vid_tables[vid] += float(row['end']) - float(row['start'])
    
    vid_duration = np.array(list(vid_tables.values()))
    print(vid_duration.sum())
   


