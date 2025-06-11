import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from backbone.resnet50_v2.resnet50_finetune import PhaseResNet50Model
from backbone.resnet50_v2.dataset import get_resnet50_transform

def feature_extraction(annotation_dir, image_dir, output_dir, model_dir):
    os.makedirs(output_dir, exist_ok=True)
    # load annotations
    df = pd.read_csv(os.path.join(annotation_dir, "image_annotations.csv"))
    
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # build model
    model = PhaseResNet50Model(pretrained=True)
    ckpt = torch.load(os.path.join(model_dir, "resnet50_finetune.pth"), map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()

    transform = get_resnet50_transform()

    # We'll collect a feature_path for each row
    df["feature_path"] = ""

    # Group by video_id so we can batch all its frames at once
    for video_id, grp in tqdm(df.groupby("video_id"), desc="Videos"):
        # sort by frame_idx
        grp = grp.sort_values("frame_idx")
        
        imgs = []
        valid_idxs = []
        for idx, row in grp.iterrows():
            if row["augment"]:
                continue
            split = row["split"]
            phase = row["phase"]
            fname = row["image_name"]
            img_path = os.path.join(image_dir, split, f"phase_{phase:02d}", fname)
            if not os.path.isfile(img_path):
                print(f"⚠️ Missing: {img_path}")
                continue
            
            img = Image.open(img_path).convert("RGB")
            imgs.append(transform(img))
            valid_idxs.append(idx)

        if len(imgs) == 0:
            continue

        # batch all frames for this video
        batch = torch.stack(imgs, dim=0).to(device)       # [T, 3, H, W]
        with torch.no_grad():
            feats, _ = model(batch)                       # feats: [T, 2048]
        
        feats = feats.cpu().numpy().transpose(1, 0)       # [2048, T]
        out_path = os.path.join(output_dir, f"{video_id}.npy")
        np.save(out_path, feats)

        # assign this same path to each row belonging to this video
        df.loc[valid_idxs, "feature_path"] = out_path

    # write updated annotations
    df.to_csv(
        os.path.join(annotation_dir, "image_annotations_with_features.csv"),
        index=False
    )
    print("✅ Feature extraction completed!")

if __name__ == "__main__":
    feature_extraction(
        annotation_dir="dataset/annotations",
        image_dir="dataset/images",
        output_dir="dataset/features",
        model_dir="backbone/resnet50_v2/models"
    )
