import json
import os
import csv
import math
import random
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchcodec.decoders import VideoDecoder

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# ───────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────

ANNOT_CSV        = "dataset/annotations/APTOS_train-val_annotation.csv"
VIDEO_DIR        = "dataset/videos"
PHASE_QUOTA_CSV  = "dataset/annotations/phase_quota_config.csv"   # output from the previous step
OUTPUT_IMG_DIR   = "dataset/train_images"            # root directory to save images
MAPPING_CSV      = "dataset/annotations/train_image_mapping.csv"

# Augmentation pipeline for "rare" frames (applied only when is_aug=True)
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomRotation(15),
])

# Ensure the top-level output directory exists
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# ───────────────────────────────────────────────
# STEP 1: LOAD QUOTAS PER PHASE
# ───────────────────────────────────────────────

quota_df = pd.read_csv(PHASE_QUOTA_CSV)
# Expect columns: phase_id,duration_s,available_frames,quota,use_weight
quotas = {int(row.phase_id): int(row.quota) for _, row in quota_df.iterrows()}

# ───────────────────────────────────────────────
# STEP 2: READ ANNOTATIONS & GROUP BY PHASE
# ───────────────────────────────────────────────

annots = {}  # phase_id -> list of (video_id, start_time_s, end_time_s)
with open(ANNOT_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("split", "").lower() != "train":
            continue
        vid = row["video_id"]
        phase = int(row["phase_id"])
        start = float(row["start"])
        end   = float(row["end"])
        annots.setdefault(phase, []).append((vid, start, end))

# ───────────────────────────────────────────────
# STEP 3: PRELOAD VIDEO METADATA (fps → use to convert time → frames)
# ───────────────────────────────────────────────

video_ids = {vid for segments in annots.values() for (vid, _, _) in segments}
video_meta = {}  # video_id -> {"fps": ..., "total_frames": ...}
for vid in video_ids:
    path = os.path.join(VIDEO_DIR, vid + ".mp4")
    if not os.path.isfile(path):
        continue
    reader = VideoDecoder(path)
    fps = reader.metadata.average_fps or 25.0
    total_frames = reader.metadata.num_frames
    video_meta[vid] = {"fps": fps, "total_frames": total_frames}
    reader.close()

# metadata
with open(os.path.join(OUTPUT_IMG_DIR, "metadata.json"), "w") as f:
    json.dump(video_meta, f)

# ───────────────────────────────────────────────
# STEP 4: BUILD SAMPLE LIST (video_id, frame_idx, phase_id, is_aug)
# ───────────────────────────────────────────────

all_samples = []
for phase, segments in annots.items():
    # 4.1 Build per-segment "frame intervals" and total available frame count
    seg_infos = []
    total_available = 0
    for (vid, start_s, end_s) in segments:
        meta = video_meta.get(vid)
        if meta is None:
            continue
        fps = meta["fps"]
        # Convert continuous times → integer frame indices
        f0 = int(math.ceil(start_s * fps))
        f1 = int(min(meta["total_frames"] - 1, math.floor(end_s * fps)))
        if f1 < f0:
            continue
        length = f1 - f0 + 1
        seg_infos.append({"vid": vid, "f0": f0, "f1": f1, "length": length})
        total_available += length

    quota = quotas.get(phase, 0)

    # 4.2 CASE A: "Rare" class (total_available ≤ quota) → use all and augment
    if total_available <= quota:
        base_list = []
        for seg in seg_infos:
            for idx in range(seg["f0"], seg["f1"] + 1):
                base_list.append((seg["vid"], idx, False))
        missing = quota - len(base_list)
        samples_phase = base_list.copy()
        if missing > 0 and base_list:
            # Use numpy's random choice without replacement for better randomization
            indices = np.random.choice(len(base_list), size=missing, replace=True)
            for idx in indices:
                vid, frame_idx, _ = base_list[idx]
                samples_phase.append((vid, frame_idx, True))

    # 4.3 CASE B: "Medium/Popular" (total_available > quota) → TSN-style sampling
    else:
        # Compute a floating "ideal" share of quota for each segment
        float_shares = [(seg["length"] / total_available * quota) for seg in seg_infos]
        int_shares   = [math.floor(f) for f in float_shares]
        remainders   = [f - i for f, i in zip(float_shares, int_shares)]
        used = sum(int_shares)
        leftover = quota - used
        # Distribute leftover according to largest fractional remainders
        for _ in range(leftover):
            idx_max = max(range(len(remainders)), key=lambda i: remainders[i])
            int_shares[idx_max] += 1
            remainders[idx_max] = 0.0

        samples_phase = []
        for seg, share in zip(seg_infos, int_shares):
            vid, f0, f1, length = seg["vid"], seg["f0"], seg["f1"], seg["length"]
            if share >= length:
                # If share ≥ available frames, just take all original
                for idx in range(f0, f1 + 1):
                    samples_phase.append((vid, idx, False))
            else:
                # Uniformly sample exactly `share` frames from [f0 .. f1]
                for j in range(share):
                    pos = f0 + int((j + 0.5) * length / share)
                    pos = min(pos, f1)
                    samples_phase.append((vid, pos, False))

    # 4.4 Add (video_id, frame_idx, phase_id, is_aug) to global list
    for vid, idx, is_aug in samples_phase:
        all_samples.append({
            "video_id": vid,
            "frame_idx": idx,
            "phase": phase,
            "augment": is_aug
        })

# Save all_samples to json
with open(os.path.join(OUTPUT_IMG_DIR, "all_samples.json"), "w") as f:
    json.dump(all_samples, f)

# ───────────────────────────────────────────────
# STEP 5: SAVE IMAGES & WRITE MAPPING CSV
# ───────────────────────────────────────────────

# We assign a global counter to name each image uniquely
mapping_rows = []
img_counter = 1

# Keep one VideoDecoder per video to accelerate repeated frame fetches
decoders = {}

for sample in all_samples:
    vid = sample["video_id"]
    idx = sample["frame_idx"]
    phase = sample["phase"]
    is_aug = sample["augment"]

    # Open (or reuse) the VideoDecoder for this video
    if vid not in decoders:
        path = os.path.join(VIDEO_DIR, vid + ".mp4")
        decoders[vid] = VideoDecoder(path)
    reader = decoders[vid]

    # Decode the requested frame
    frame_obj = reader.get_frame_at(idx)
    tensor_frame = frame_obj.data            # Torch tensor (C,H,W)
    timestamp = float(frame_obj.pts_seconds)

    # Convert to a PIL image
    pil_img = to_pil_image(tensor_frame)

    # If flagged for augmentation, apply the augment pipeline
    if is_aug:
        pil_img = augment_transform(pil_img)

    # Determine filename & folder for this image
    img_name = f"{vid}_{img_counter:06d}.png"
    save_dir = os.path.join(OUTPUT_IMG_DIR, f"phase_{phase:02d}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, img_name)

    # Save the PNG
    pil_img.save(save_path)

    # Record the mapping entry
    mapping_rows.append({
        "video_id": vid,
        "image_id": img_name,
        "phase_id": phase,
        "timestamp": round(timestamp, 4),
    })

    img_counter += 1

# Close all decoders
for dec in decoders.values():
    dec.close()

# Write out the mapping CSV
with open(MAPPING_CSV, "w", newline="") as f:
    fieldnames = ["video_id", "image_id", "phase_id", "timestamp"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in mapping_rows:
        writer.writerow(row)

print(f"Done! Saved {len(mapping_rows)} images and mapping to {MAPPING_CSV}.")
