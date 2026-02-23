import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--video_id", default="rav4_video")
    p.add_argument("--seconds_per_frame", type=int, required=True)
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out_parquet", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    frames = sorted(Path(args.frames_dir).glob("*.jpg"))
    if not frames:
        raise SystemExit("No frames found.")

    model = YOLO(args.model)

    rows = []

    for frame_index, img_path in enumerate(tqdm(frames, desc="Running detection")):
        timestamp_sec = frame_index * args.seconds_per_frame

        result = model.predict(source=str(img_path), conf=args.conf, verbose=False)[0]

        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clses):
            rows.append({
                "video_id": args.video_id,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "class_label": names[int(cls_id)],
                "x_min": float(x1),
                "y_min": float(y1),
                "x_max": float(x2),
                "y_max": float(y2),
                "confidence_score": float(c),
                "model_name": args.model
            })

    df = pd.DataFrame(rows)
    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)

    print(f"Saved {len(df)} detections to {args.out_parquet}")


if __name__ == "__main__":
    main()