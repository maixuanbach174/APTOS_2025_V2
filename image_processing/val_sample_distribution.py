import csv, math
import pandas as pd

# Thông số
ANNOT = "dataset/annotations/APTOS_train-val_annotation.csv"
VIDEO_DIR = "dataset/videos"
FPS = 25
NUM_CLASSES = 35

# Ngưỡng
THRESH_VERY_RARE = int(500 / 4)     # <500 khung → không ép augmentation quá nhiều
THRESH_RARE = int(1000 / 4)          # <1000 khung thì xoay vào nhóm “hiếm vừa”    # target ảnh cho nhóm hiếm vừa
TRHESH_MEDIUM = int(1500 / 4)
TRHESH_POPULAR = int(2000 / 4)          # max ảnh cho nhóm phổ biến

duration_sums = [0.0]*NUM_CLASSES
with open(ANNOT, newline='') as f:
    rd = csv.DictReader(f)
    for r in rd:
        vid = r['video_id']
        if r.get('split','train').lower() != 'val': 
            continue
        c = int(r['phase_id'])
        s = float(r['start']); e = float(r['end'])
        duration_sums[c] += max(0.0, e - s)

available_frames = [ int(math.floor(d*FPS)) for d in duration_sums ]

quotas = [0]*NUM_CLASSES
use_weight = [False]*NUM_CLASSES
type = ["medium"] * NUM_CLASSES

for c in range(NUM_CLASSES):
    n_fr = available_frames[c]
    if n_fr == 0:
        quotas[c] = 0
        type[c] = "no_video"
        use_weight[c] = False
    elif n_fr < THRESH_VERY_RARE:
        quotas[c] = THRESH_VERY_RARE    
        use_weight[c] = True 
        type[c] = "very_rare"
    elif n_fr < THRESH_RARE:
        quotas[c] = THRESH_RARE
        type[c] = "rare"
        use_weight[c] = False
    else:
        quotas[c] = TRHESH_MEDIUM
        type[c] = "medium"
        use_weight[c] = False
    if n_fr > TRHESH_POPULAR:
        quotas[c] = int(1750 / 4)
        type[c] = "popular"
        use_weight[c] = False

print(f"{'Phase':>5} | Dur(s) | Avail_Frames | Quota | Use_Weight")
print("-"*50)
for c in range(NUM_CLASSES):
    print(f"{c:>5} | {duration_sums[c]:>6.1f} | {available_frames[c]:>12d} | {type[c]:>10} |"
          f"{quotas[c]:>5d} | {str(use_weight[c]):>10}")

df = pd.DataFrame({
    "phase_id": list(range(NUM_CLASSES)),
    "duration_s": [round(d, 2) for d in duration_sums],
    "available_frames": available_frames,
    "type": type,
    "quota": quotas,
    "use_weight": use_weight
})
df.to_csv("dataset/annotations/phase_quota_config_val.csv", index=False)
print(df["quota"].sum())
