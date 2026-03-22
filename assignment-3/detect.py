import cv2
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

FRAMES_DIR     = Path("frames")
DETECTIONS_DIR = Path("detections")
MODEL_PATH     = "best.pt"
CONFIDENCE     = 0.3

model = YOLO(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")
print(f"Classes: {model.names}")

for video_frames_dir in sorted(FRAMES_DIR.iterdir()):
    if not video_frames_dir.is_dir():
        continue

    video_name = video_frames_dir.name
    out_dir = DETECTIONS_DIR / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(video_frames_dir.glob("*.jpg"))
    print(f"\nProcessing {video_name}: {len(frame_paths)} frames")

    detection_count = 0
    for frame_path in tqdm(frame_paths):
        results = model(frame_path, conf=CONFIDENCE, verbose=False)

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                # Read the frame
                img = cv2.imread(str(frame_path))

                # Draw each bounding box
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    label = f"drone {conf:.2f}"

                    # Draw rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)

                    cv2.putText(img, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                cv2.imwrite(str(out_dir / frame_path.name), img)
                detection_count += 1
                break

    print(f"Saved {detection_count}/{len(frame_paths)} frames with detections")

print("\nDetection complete!")
print(f"Results saved to: {DETECTIONS_DIR}")