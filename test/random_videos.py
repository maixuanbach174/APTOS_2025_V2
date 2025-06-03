import csv
import random

train_videos = set()
val_videos = set()
with open("dataset/annotations/APTOS_train-val_annotation.csv", "r", newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("split", "train").lower() == 'train':
            train_videos.add(row.get("video_id"))
        else:
            val_videos.add(row.get("video_id"))

train_videos = list(train_videos)
val_videos = list(val_videos)
random.shuffle(train_videos)
random.shuffle(val_videos)

print(train_videos[:16])
print(val_videos[:4])

# ['case_1603', 'case_0888', 'case_0152', 'case_0342', 'case_0429', 'case_0984', 'case_0735', 'case_1119', 'case_1905', 'case_1884', 'case_0183', 'case_1721', 'case_1315', 'case_1555', 'case_1210', 'case_0043']
#['case_1054', 'case_0998', 'case_0776', 'case_1492']

# dict = {0: 1750, 1: 1750, 2: 1750, 3: 1750, 4: 1750, 5: 1750, 6: 1750, 7: 1750, 8: 1000, 9: 1750, 10: 1750, 11: 500, 12: 1750, 13: 1750, 14: 1750, 15: 500, 16: 1750, 17: 1750, 18: 1750, 19: 1750, 20: 1750, 21: 1000, 22: 1750, 23: 500, 24: 1750, 25: 1750, 26: 1750, 27: 1750, 28: 1750, 29: 1750, 30: 1750, 31: 1750, 32: 1750, 33: 1750, 34: 1750}
# values = list(dict.values())
# print(sum(values))