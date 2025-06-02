import os
import glob
import csv
import math

ANNOT_CSV = "dataset/annotations/APTOS_train-val_annotation.csv"
VIDEO_DIR  = "dataset/videos"
FPS        = 25           
NUM_CLASSES = 35

MIN_FRAMES = 500
MAX_FRAMES = 1500
TARGET     = 1000

mp4_paths = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
valid_ids = { os.path.splitext(os.path.basename(p))[0] for p in mp4_paths }

duration_sums = [0.0] * NUM_CLASSES
with open(ANNOT_CSV, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid = row['video_id']
        if vid not in valid_ids:
            continue
        if row.get('split','train').lower() != 'train':
            continue
        phase = int(row['phase_id'])
        start = float(row['start'])
        end   = float(row['end'])
        dur = max(0.0, end - start)
        duration_sums[phase] += dur

# 4) Quy đổi total duration -> approx. số khung hình
available_frames = []
for c in range(NUM_CLASSES):
    n = int(math.floor(duration_sums[c] * FPS))
    available_frames.append(n)

# 5) Thiết lập quota theo ba trường hợp
quotas = {}
for c in range(NUM_CLASSES):
    n_avail = available_frames[c]
    if n_avail < MIN_FRAMES:
        # quá hiếm, ta vẫn gán quota = 1000 (sau đó sẽ augment)
        q = TARGET
    elif n_avail <= MAX_FRAMES:
        # vừa đủ hoặc chừng, lấy đúng số khung có sẵn
        q = n_avail
    else:
        # quá nhiều, chỉ lấy tối đa 1500 để còn biến thể (TSN, random)
        q = MAX_FRAMES

    quotas[c] = q

# 6) In kết quả ra, hoặc luu vào file
print("Phase ID | total_dur(s) | avail_frames | quota_to_sample")
print("-" * 50)
for c in range(NUM_CLASSES):
    print(f"{c:>7} |  {duration_sums[c]:6.2f}       |  {available_frames[c]:6d}     |  {quotas[c]:6d}")

# Nếu muốn lưu vào CSV cho giai đoạn preprocessing sau này:
import pandas as pd
df = pd.DataFrame({
    "phase_id": list(range(NUM_CLASSES)),
    "total_duration": duration_sums,
    "available_frames": available_frames,
    "quota": [quotas[c] for c in range(NUM_CLASSES)]
})
df.to_csv("phase_sampling_quotas.csv", index=False)
