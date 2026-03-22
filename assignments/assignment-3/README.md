# UAV Drone Detection and Tracking

## Output Tracking Videos

- **Video 1:** https://www.youtube.com/watch?v=iF_yNeWu_cE
- **Video 2:** https://www.youtube.com/watch?v=ntMsV9wvUm8

---

## Hugging Face Dataset

Detection frames are available at:
https://huggingface.co/datasets/saanvielaty/drone-detections

569 frames (495 from video 1, 74 from video 2) stored in Parquet format.

### Dataset Choice
I used the [Drone Detection dataset](https://universe.roboflow.com/drone-detection-g4d3g/drone-detection-a1tsf) from Roboflow Universe, containing 9,972 labeled images of drones captured from ground-level and side-view perspectives. This dataset was chosen because:
- It detects the drone itself (not aerial imagery from a drone)
- Single class: `drones`
- Large and diverse — various backgrounds, lighting conditions, and drone types
- Available in YOLOv8 format (COCO/YOLO compatible)

### Detector Configuration
- **Model:** YOLOv8s (small) pretrained on COCO, fine-tuned on drone dataset
- **Training:** 20 epochs, batch size 16, image size 640x640, GPU (Tesla T4 via Google Colab)
- **Optimizer:** AdamW (lr=0.002)
- **Confidence threshold:** 0.3
- **Results:** 495/828 frames detected in video 1, 74/2580 frames in video 2

---

## Task 2: Kalman Filter Tracking

### Filter Design
I used the `filterpy` library with a constant velocity motion model.

**State vector:** `[x, y, vx, vy]`
- `x, y` — bounding box center position in pixels
- `vx, vy` — velocity in pixels per frame

**Matrices:**
- State transition matrix `F`: constant velocity model
- Measurement matrix `H`: observes `x, y` only (position, not velocity)
- Measurement noise `R`: scaled by 10 (accounts for detector jitter)
- Process noise `Q`: scaled by 0.1 (smooth motion assumption)
- Initial covariance `P`: scaled by 100 (high initial uncertainty)

### Predict / Update Cycle
For each frame:
1. **Predict** — `kf.predict()` propagates state forward using `F`
2. **Update** — if a detection exists, `kf.update()` corrects the prediction
3. **Coast** — if no detection, the filter predicts for up to 10 consecutive frames

### Handling Missing Detections
When the detector misses the drone (e.g. due to motion blur or small size), the Kalman filter continues predicting the drone's position using its velocity estimate for up to `MAX_MISSING=10` frames. These predicted positions are shown as orange circles in the output video.

---

## Failure Cases
- **Video 2** has a low detection rate (74/2580 frames ~3%) because the drone is very small and far away for most of the video. The Kalman filter compensates by predicting across gaps but loses the track when the drone is gone for more than 10 consecutive frames.
- Fast direction changes cause the constant velocity model to lag behind briefly before the update step corrects it.
- Bright sky backgrounds occasionally cause false negatives at the detector level.

```