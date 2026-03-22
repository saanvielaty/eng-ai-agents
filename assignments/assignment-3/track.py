import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from tqdm import tqdm

FRAMES_DIR      = Path("frames")
OUTPUT_DIR      = Path("output_videos")
MODEL_PATH      = "best.pt"
CONFIDENCE      = 0.3
MAX_MISSING     = 10   

# Kalman Filter Setup
def make_kalman():
    """
    State vector: [x, y, vx, vy]
    x, y   = bounding box center position
    vx, vy = velocity
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)

    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)

    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]], dtype=float)

    kf.R *= 10

    kf.Q *= 0.1

    kf.P *= 100

    return kf

def draw_trajectory(img, trajectory, color=(0, 255, 255)):
    for i in range(1, len(trajectory)):
        pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
        pt2 = (int(trajectory[i][0]),   int(trajectory[i][1]))
        cv2.line(img, pt1, pt2, color, 2)

def process_video(video_frames_dir, output_path, model):
    frame_paths = sorted(video_frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"No frames found in {video_frames_dir}")
        return

    sample = cv2.imread(str(frame_paths[0]))
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, 5, (w, h))

    kf = make_kalman()
    initialized = False
    missing = 0
    trajectory = []
    frames_written = 0

    print(f"\nTracking {video_frames_dir.name}: {len(frame_paths)} frames")

    for frame_path in tqdm(frame_paths):
        img = cv2.imread(str(frame_path))
        results = model(frame_path, conf=CONFIDENCE, verbose=False)

        detected = False
        cx, cy = None, None
        box_coords = None

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                best = result.boxes[result.boxes.conf.argmax()]
                x1, y1, x2, y2 = map(int, best.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                box_coords = (x1, y1, x2, y2)
                detected = True
                break

        if detected:
            if not initialized:
                kf.x = np.array([[cx], [cy], [0.], [0.]], dtype=float)
                initialized = True
            else:
                kf.predict()
                kf.update(np.array([[cx], [cy]]))

            missing = 0
            est_x, est_y = float(kf.x[0][0]), float(kf.x[1][0])
            trajectory.append((est_x, est_y))

            x1, y1, x2, y2 = box_coords
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            conf = float(results[0].boxes.conf[results[0].boxes.conf.argmax()])
            cv2.putText(img, f"drone {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            draw_trajectory(img, trajectory)

            writer.write(img)
            frames_written += 1

        elif initialized and missing < MAX_MISSING:
            kf.predict()
            missing += 1
            est_x, est_y = float(kf.x[0][0]), float(kf.x[1][0])
            trajectory.append((est_x, est_y))

            cv2.circle(img, (int(est_x), int(est_y)), 8, (0, 165, 255), -1)
            cv2.putText(img, "predicted", (int(est_x) + 10, int(est_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # Draw trajectory
            draw_trajectory(img, trajectory)

            writer.write(img)
            frames_written += 1

    writer.release()
    print(f"Wrote {frames_written} frames to {output_path}")

OUTPUT_DIR.mkdir(exist_ok=True)
model = YOLO(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")

for video_frames_dir in sorted(FRAMES_DIR.iterdir()):
    if not video_frames_dir.is_dir():
        continue
    output_path = OUTPUT_DIR / f"{video_frames_dir.name}_tracked.mp4"
    process_video(video_frames_dir, output_path, model)

print("\nTracking complete!")
print(f"Output videos saved to: {OUTPUT_DIR}")