import csv

with open("dataset/annotations/APTOS_train-val_annotation.csv", "r", newline='') as f:
    reader = csv.DictReader(f)
    counter = 0
    val_videos = set()
    for row in reader:
        if row.get("split", "train").lower() == 'train':
            val_videos.add(row.get("video_id"))

print(val_videos)