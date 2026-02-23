import argparse
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from ultralytics import YOLO


def merge_segments(timestamps, step=1, max_gap=None):
    """
    Merge sorted timestamps into contiguous segments.
    step: expected step between samples (seconds_per_frame)
    max_gap: allow small gaps (in seconds); default = step (strict contiguity)
    Returns list of dicts: {start_timestamp, end_timestamp, n_supporting_detections}
    """
    if not timestamps:
        return []
    if max_gap is None:
        max_gap = step

    segs = []
    start = prev = timestamps[0]
    n = 1

    for t in timestamps[1:]:
        if t - prev <= max_gap:
            prev = t
            n += 1
        else:
            segs.append(
                {
                    "start_timestamp": int(start),
                    "end_timestamp": int(prev),
                    "n_supporting_detections": int(n),
                }
            )
            start = prev = t
            n = 1

    segs.append(
        {
            "start_timestamp": int(start),
            "end_timestamp": int(prev),
            "n_supporting_detections": int(n),
        }
    )
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections_parquet", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seconds_per_frame", type=int, default=1)
    ap.add_argument("--query_index", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--max_gap", type=int, default=2, help="merge gaps up to N seconds into one segment")
    ap.add_argument("--top_k", type=int, default=20, help="print top K segments")
    args = ap.parse_args()

    # Load detections index (Parquet from video inference)
    df = pd.read_parquet(args.detections_parquet)

    # Load query dataset (HF)
    ds = load_dataset("aegean-ai/rav4-exterior-images", split="train")
    ex = ds[args.query_index]
    img = ex["image"]  # PIL image

    print(f"Query index: {args.query_index} | timestamp_sec (metadata): {ex['timestamp_sec']}")

    # Detect parts in query image
    model = YOLO(args.model)
    r = model.predict(source=img, conf=args.conf, verbose=False)[0]

    if r.boxes is None or len(r.boxes) == 0:
        print("No detections in query image at this confidence. Try --conf 0.25 or a different --query_index.")
        return

    names = r.names
    cls_ids = r.boxes.cls.cpu().numpy().astype(int).tolist()
    confs = r.boxes.conf.cpu().numpy().tolist()

    # Choose retrieval label by total confidence (more stable than raw counts)
    label_score = defaultdict(float)
    label_counts = defaultdict(int)

    for cid, conf in zip(cls_ids, confs):
        lbl = names[cid]
        label_score[lbl] += float(conf)
        label_counts[lbl] += 1

    chosen_label = sorted(
        label_score.items(),
        key=lambda x: (-x[1], -label_counts[x[0]], x[0])
    )[0][0]

    print("Query label counts:", dict(sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))))
    print("Query label scores:", dict(sorted(label_score.items(), key=lambda x: (-x[1], x[0]))))
    print("Chosen class_label for retrieval:", chosen_label)

    # Retrieve timestamps where chosen label appears in video detections
    ts = sorted(df.loc[df["class_label"] == chosen_label, "timestamp_sec"].astype(int).unique().tolist())
    if not ts:
        print("No matches for chosen label in video index.")
        return

    segs = merge_segments(ts, step=args.seconds_per_frame, max_gap=args.max_gap)

    # Sort by best evidence first (most supporting detections, then longest)
    segs.sort(key=lambda s: (-s["n_supporting_detections"], -(s["end_timestamp"] - s["start_timestamp"])))

    print("\nstart_timestamp,end_timestamp,class_label,number_of_supporting_detections,youtube_verify")
    for s in segs[: args.top_k]:
        a = s["start_timestamp"]
        b = s["end_timestamp"]
        n = s["n_supporting_detections"]
        url = f"https://www.youtube.com/embed/YcvECxtXoxQ?start={a}&end={b}"
        print(f"{a},{b},{chosen_label},{n},{url}")

    print(f"\nTotal segments: {len(segs)}")


if __name__ == "__main__":
    main()