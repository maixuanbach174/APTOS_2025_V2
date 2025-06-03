import pandas as pd

df = pd.read_csv("dataset/annotations/samples_mapping.csv")

count_vid_id_augment = {}
for _, s in df.iterrows():  
    if s["augment"] == True:
        count_vid_id_augment[s["phase"]] = count_vid_id_augment.get(s["phase"], 0) + 1

print(count_vid_id_augment)