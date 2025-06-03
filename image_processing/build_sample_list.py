import glob
import json
import os
import csv
import math
import random
import pandas as pd
from torchvision import transforms

from torchcodec.decoders import VideoDecoder

# ───────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────

ANNOT_CSV        = "dataset/annotations/APTOS_train-val_annotation.csv"
VIDEO_DIR        = "dataset/videos"
PHASE_QUOTA_CSV  = "dataset/annotations/phase_quota_config.csv"   # output from the previous step
OUTPUT_IMG_DIR   = "dataset/train_images"            # root directory to save images
SAMPLES_CSV      = "dataset/annotations/samples_mapping.csv"      # output samples mapping file
BUFFER_FRAMES    = 2  # Số frame buffer ở đầu và cuối mỗi segment cho medium/popular

# Ensure the top‐level output directory exists
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# ───────────────────────────────────────────────
# STEP 1: LOAD QUOTAS PER PHASE
# ───────────────────────────────────────────────

quota_df = pd.read_csv(PHASE_QUOTA_CSV)
# Expect columns: phase_id,duration_s,available_frames,quota,use_weight
quotas = {int(row.phase_id): int(row.quota) for _, row in quota_df.iterrows()}
print(quotas)

# skip video not in directory
video_ids = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))}

# ───────────────────────────────────────────────
# STEP 2: READ ANNOTATIONS & GROUP BY PHASE
# ───────────────────────────────────────────────

annots = {}  # phase_id -> list of (video_id, start_time_s, end_time_s)
with open(ANNOT_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["video_id"] not in video_ids:
            continue
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
    del reader

all_samples = []
for phase, segments in annots.items():
    # 4.1 Build per‐segment "frame intervals" and total available frame count
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
        
        # Cycle through the base list to fill the missing quota
        if missing > 0 and base_list:
            base_list_size = len(base_list)
            for i in range(missing):
                vid, idx, _ = base_list[i % base_list_size]  # Cycle through the list
                samples_phase.append((vid, idx, True))

    # 4.3 CASE B: "Medium/Popular" (total_available > quota) → TSN‐style sampling
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
            
            # Thêm buffer frames cho medium/popular segments
            safe_f0 = f0 + BUFFER_FRAMES
            safe_f1 = f1 - BUFFER_FRAMES
            safe_length = safe_f1 - safe_f0 + 1
            
            if share >= length:
                # If share ≥ available frames, just take all original but respect buffer
                for idx in range(safe_f0, safe_f1 + 1):
                    samples_phase.append((vid, idx, False))
            else:
                # Uniformly sample exactly `share` frames from [safe_f0 .. safe_f1]
                for j in range(share):
                    pos = safe_f0 + int((j + 0.5) * safe_length / share)
                    pos = min(max(pos, safe_f0), safe_f1)  # đảm bảo nằm trong vùng an toàn
                    samples_phase.append((vid, pos, False))

    # 4.4 Add (video_id, frame_idx, phase_id, is_aug) to global list
    for vid, idx, is_aug in samples_phase:
        all_samples.append({
            "video_id": vid,
            "frame_idx": idx,
            "phase": phase,
            "augment": is_aug
        })

# Save samples to CSV
with open(SAMPLES_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["video_id", "frame_idx", "phase", "augment"])
    writer.writeheader()
    for sample in all_samples:
        writer.writerow(sample)

print(f"Saved {len(all_samples)} samples to {SAMPLES_CSV}")

# Save metadata
with open(os.path.join(OUTPUT_IMG_DIR, "metadata.json"), "w") as f:
    json.dump(video_meta, f)