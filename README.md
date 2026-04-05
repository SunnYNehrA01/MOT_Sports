# MOT Sports :- Multiple Object Tracking in Sports

## Project Overview
This project is a high-performance **Multi-Object Tracking (MOT)** system designed specifically for the chaotic, high-contact environment of broadcast sports (Basketball, Football, etc.). Unlike standard trackers that fail when players overlap, this system implements **Temporal Identity Persistence** to ensure a player's ID remains "sticky" even after long occlusions.

---

## Thought Process
The primary challenge in sports tracking is **occlusion**. In basketball, players constantly screen each other, causing bounding boxes to merge. 
1. **The Goal**: Create a tracker that doesn't just "see" boxes but "remembers" visual identities.
2. **The Solution**: Move beyond simple IoU (Intersection over Union) matching. By combining motion prediction (Kalman Filters) with visual fingerprints (Re-ID) and logical constraints (Team Color Gating), we create a three-stage association process that survives player crossovers.

---

## System Architecture
The pipeline follows a modular "Detection-then-Tracking" architecture:

1. **Detection Layer**: YOLOv11 (Small/Medium/Nano) extracts player and ball coordinates.
2. **Feature Extraction**: A ResNet-18 backbone generates 128-dimensional "Appearance Embeddings" for each player.
3. **Motion Compensation (CMC)**: Uses OpenCV Global Motion Estimation to correct for camera pans and zooms.
4. **The ByteTrack Engine**: 
    * **Stage 1**: Matches high-confidence detections using a weighted cost of IoU and Cosine Similarity.
    * **Stage 2**: Recovers occluded players using low-confidence detections and Kalman Filter predictions.
    * **Stage 3 (The Fix)**: Uses a "Lost Track Buffer" to re-identify players who separated after a cluster using their visual fingerprint.

---

## Project Structure
```text
MOT-Sports-App/
├── core/
│   ├── engine.py           # The "Heart": Coordinates detection, tracking, and video I/O
│   ├── detector.py         # YOLOv11 Inference wrapper
│   └── reid.py             # ResNet-18 Feature extraction for identity persistence
├── modules/
│   ├── tracker.py          # Custom ByteTrack implementation with Stage 3 Recovery
│   └── kalman.py           # Linear Kalman Filter for state prediction
├── utils/
│   ├── video_utils.py      # Frame processing and FFmpeg encoding helpers
│   └── drawing_utils.py    # Logic for trails, bounding boxes, and ID labels
├── app.py                  # Streamlit Bento-Box UI
└── requirements.txt        # Production dependencies
```

---

## Components & Why They Are Used
| Component | Purpose | Why This? |
| :--- | :--- | :--- |
| **YOLOv11** | Object Detection | State-of-the-art speed/accuracy trade-off for real-time sports. |
| **Kalman Filter** | Motion Prediction | Predicts where a player *should* be if they disappear behind another. |
| **ResNet-18 (Re-ID)** | Visual Fingerprinting | Distinguishes between players who look similar from a distance. |
| **FFmpeg** | Video Encoding | Converts raw OpenCV output to H.264 so it plays in web browsers. |
| **Streamlit** | Dashboard UI | Fast, Python-native way to build a professional-grade analysis tool. |

---

## Problems Encountered & Solutions

### 1. The "ID Swap" Problem
* **Problem**: When two players overlapped and separated, their IDs would often flip.
* **Tackle**: Implemented **Stage 3 Association**. If a track is lost, we keep it in a "Lost Buffer" for 60 frames. When a person reappears, we compare their visual fingerprint against the buffer to restore their original ID.

### 2. Browser Playback Failure
* **Problem**: The output videos from OpenCV would download but wouldn't play in Chrome or Edge.
* **Tackle**: Integrated an **FFmpeg post-processing step** that re-encodes the `.mp4` into H.264 format with a YUV420p pixel format.

---

## Evaluation & Comparison Logic
The system allows for the comparison of three distinct YOLOv11 variants:
* **YOLO11-Nano**: Optimized for CPU/Mobile; lowest latency, slightly lower precision on small objects (e.g., the ball).
* **YOLO11-Small**: The "Sweet Spot"; balanced for real-time tracking with high identity stability.
* **YOLO11-Medium**: Maximum Accuracy; best for crowded "Paint" areas in basketball but requires higher GPU memory.

---

