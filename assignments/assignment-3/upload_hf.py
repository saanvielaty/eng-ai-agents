import os
from pathlib import Path
from datasets import Dataset
import pandas as pd

DETECTIONS_DIR = Path("detections")
HF_TOKEN       = "huggingfacetoken"   
HF_REPO        = "repo name"  

records = []
for video_dir in sorted(DETECTIONS_DIR.iterdir()):
    if not video_dir.is_dir():
        continue
    for img_path in sorted(video_dir.glob("*.jpg")):
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        records.append({
            "video":    video_dir.name,
            "filename": img_path.name,
            "image":    img_bytes,
        })

print(f"Total detection frames collected: {len(records)}")

df = pd.DataFrame(records)
dataset = Dataset.from_pandas(df)

print("Pushing to Hugging Face...")
dataset.push_to_hub(HF_REPO, token=HF_TOKEN)
print(f"Done! Dataset available at: https://huggingface.co/datasets/{HF_REPO}")