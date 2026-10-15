# ProYOLO DefOps • Real-Time Physical Adversarial Patch Defense & Surveillance Console

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX%20CPU-FF6F00.svg)](https://github.com/ultralytics/ultralytics)
[![Tests](https://img.shields.io/badge/Tests-67%20Passed-emerald.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**ProYOLO** is an end-to-end physical adversarial defense framework designed for CCTV and smart surveillance infrastructures. It intercepts printed optical adversarial patches (designed to fool deep object detectors like YOLOv8n) in real time using CPU-efficient dual temporal-spatial feature extraction and lightweight machine learning classification without requiring GPU acceleration.

---

## Key Features

- **Real-Time Video Processing**: Active OpenCV ingestion pipeline with laptop webcam (`cv2.VideoCapture(0)`), RTSP network streams, or uploaded `.mp4`/`.avi` video files.
- **CPU-Optimized YOLOv8 Object Detection**: High-throughput person proposal generation using ONNX Runtime with multi-threaded sequential execution ($30+\text{ FPS}$ on multi-core CPUs).
- **ByteTrack Tracking Memory**: 10-frame sliding window temporal tracking with low-confidence association to maintain trajectory continuity.
- **Dual Visual & Temporal Anomaly Extraction**:
  - *Temporal Features*: Sudden confidence drops, bounding box area/aspect-ratio instability, consecutive IoU variance, and detection flicker rates.
  - *Spatial/Color Features*: Upper-torso Laplacian gradient variance/entropy and HSV color dispersion entropy.
- **Sub-2ms Attack Classification**: Supervised classifier (`RandomForest`, `HistGradientBoosting`, or `XGBoost`) delivering $<2\text{ms}$ inference per track with top-feature explainability.
- **DefOps Web Dashboard**:
  - Motion-JPEG live video feed (`/api/v1/video-feed`) with dynamic bounding box annotations.
  - Clear real-time classification: **`Person`** (Emerald Green) vs. **`Attacker`** (Crimson Red with threat badge).
  - Source switching toolbar: Webcam (Live), Video File Upload (`.mp4`, `.avi`), Play/Pause, and EOF Looping.
  - Live WebSocket (`/ws/alerts`) telemetry feed with forensic base64 snapshot crops and audio radar chirps.
  - SQLite database persistence for forensic auditing and incident replay.

---

## Detection & Classification Legend

| Classification | Bounding Box Color | Label Format | Description |
| :--- | :--- | :--- | :--- |
| **Person** | **Emerald Green** | `Person #<track_id> (<conf>%)` | Verified human pedestrian with normal temporal and spatial feature signatures. |
| **Attacker** | **Crimson Red** | `Attacker #<track_id> (<prob>%)` | Person carrying a physical adversarial optical patch flagged by the defense classifier. |

---

## Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │    CCTV / Webcam / Uploaded Video    │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │   Active Ingestion & Normalization   │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │   YOLOv8n ONNX Detector (384x384)    │
                      └──────────────────┬───────────────────┘
                                         │ Person Detections (Class 0)
                                         ▼
                      ┌──────────────────────────────────────┐
                      │     ByteTrack Multi-Object Tracker   │
                      │       (10-Frame Sliding Memory)      │
                      └──────────────────┬───────────────────┘
                                         │ Active Tracks
                                         ▼
                      ┌──────────────────────────────────────┐
                      │      Dual Feature Extractor          │
                      │  • Temporal: Conf Drop, IoU, Flicker │
                      │  • Visual: Laplacian & HSV Entropy   │
                      └──────────────────┬───────────────────┘
                                         │ 13-Dim Feature Vector
                                         ▼
                      ┌──────────────────────────────────────┐
                      │     Lightweight Attack Classifier    │
                      │       (RandomForest / XGBoost)       │
                      └──────────────────┬───────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
    ┌───────────────────────────────┐         ┌───────────────────────────────┐
    │     Clean Detections:         │         │     Flagged Attacks:          │
    │     Person #<id> (Green Box)  │         │     Attacker #<id> (Red Box)  │
    └───────────────┬───────────────┘         └───────────────┬───────────────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                      ┌──────────────────────────────────────┐
                      │         ProYOLO DefOps Backend       │
                      │  • MJPEG Stream: /api/v1/video-feed  │
                      │  • WebSocket Telemetry: /ws/alerts   │
                      │  • SQLite Database: surveillance.db  │
                      │  • Web Dashboard: static/index.html  │
                      └──────────────────────────────────────┘
```

---

## Directory Structure

```
ProYOLO/
├── models/
│   └── attack_classifier.joblib   # Trained physical patch defense classifier
├── src/
│   ├── classifier/
│   │   ├── attack_classifier.py   # Sub-2ms inference wrapper & feature explainability
│   │   └── train_classifier.py    # Model training, 5-fold CV & ONNX export
│   ├── detection/
│   │   ├── detector.py            # CPU-optimized YOLOv8 ONNX Runtime detector
│   │   ├── tracker.py             # ByteTrack tracker with 10-frame memory
│   │   └── feature_extractor.py   # Dual temporal and spatial/color entropy extractor
│   ├── pipeline/
│   │   ├── redis_manager.py       # Frame queue manager with fakeredis fallback
│   │   └── stream_ingestion.py    # Background video ingestion worker
│   ├── server/
│   │   ├── app.py                 # FastAPI REST API & WebSocket broadcast server
│   │   ├── database.py            # SQLAlchemy SQLite alert storage
│   │   ├── stream_engine.py       # Active OpenCV Motion-JPEG streaming engine
│   │   └── static/
│   │       ├── index.html         # DefOps real-time monitoring web console
│   │       └── app.js             # Client dashboard logic & WebSocket handling
│   ├── evaluation/
│   │   └── benchmark.py           # Physical CCTV robustness benchmarking suite
│   └── main.py                    # Unified CLI pipeline runner
├── reports/
│   └── benchmark_report.md        # CCTV robustness evaluation report
├── tests/                         # Full automated test suite (67 tests)
│   ├── test_video_endpoints.py    # Upload & MJPEG streaming endpoint tests
│   ├── test_server.py             # REST API & WebSocket broadcast tests
│   ├── test_classifier.py         # Attack classifier unit & latency tests
│   ├── test_feature_extractor.py  # Detector, tracker & feature extractor tests
│   └── test_runner_and_benchmark.py # Pipeline integration & benchmark tests
├── yolov8n.onnx                   # CPU-optimized YOLOv8n ONNX model
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- (Optional) Redis server for distributed multi-process queuing. If Redis is not installed, ProYOLO automatically falls back to an in-memory `fakeredis` queue with zero configuration required.

### 1. Clone the Repository
```bash
git clone https://github.com/VinayakKumarSingh/ProYOLO.git
cd ProYOLO
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run the Project

### Mode 1: DefOps Web Dashboard & Backend (Recommended)

Start the FastAPI application and OpenCV streaming server:

```bash
uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
- **Interactive Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Motion-JPEG Video Feed**: [http://localhost:8000/api/v1/video-feed](http://localhost:8000/api/v1/video-feed)

#### Dashboard Features:
1. **Webcam (Live)**: Click the **Webcam (Live)** button to stream directly from device `0` (built-in laptop webcam or USB camera).
2. **Upload Video**: Click **Upload Video** and select any recorded `.mp4` or `.avi` surveillance clip. The system switches processing to that file frame-by-frame with automatic looping.
3. **Playback Controls**: Toggle **Pause / Resume** or **Loop: ON/OFF** directly from the UI toolbar.
4. **Real-Time Threat Feed**: When an adversarial attacker enters the camera FOV, an alert card dynamically pops into the feed containing the track ID, attack probability, and a cropped snapshot of the patch.
5. **Forensic Modal**: Click any alert thumbnail to inspect the full forensic breakdown (bounding box coordinates, YOLO detection confidence, attacker probability, and top contributing features).

---

### Mode 2: Unified CLI Pipeline Runner (`src/main.py`)

Run the standalone ingestion $\to$ tracking $\to$ feature extraction $\to$ defense classifier pipeline from the command line:

```bash
# 1. Run on live webcam with desktop GUI window
python src/main.py --source 0

# 2. Run on a recorded video file
python src/main.py --source /path/to/surveillance_video.mp4

# 3. Headless mode (no desktop window, streams alerts directly to the FastAPI server)
python src/main.py --source 0 --headless --server-url http://localhost:8000

# 4. Custom detection threshold and camera ID
python src/main.py --source 0 --camera-id CAM-NORTH --threshold 0.70
```

#### CLI Options:
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--source` | Webcam index (`0`), video file path, or RTSP URL | `0` |
| `--model` | Path to YOLOv8 model (`.onnx` or `.pt`) | `yolov8n.onnx` |
| `--classifier` | Path to trained attack classifier | `models/attack_classifier.joblib` |
| `--camera-id` | Camera identifier string | `cam-01` |
| `--fps` | Target ingestion frame rate | `15.0` |
| `--threshold` | Attack probability threshold for alerts | `0.65` |
| `--server-url` | DefOps FastAPI server URL for alert dispatch | `http://localhost:8000` |
| `--headless` | Run without opening desktop OpenCV windows | `False` |

---

### Mode 3: Train and Benchmark Defense Classifiers

Train and benchmark `RandomForest`, `HistGradientBoosting`, and `XGBoost` classifiers across clean and adversarial feature vectors:

```bash
python src/classifier/train_classifier.py
```

This runs 5-fold cross-validation, benchmarks CPU latency per sample, saves the best model to `models/attack_classifier.joblib`, and exports an ONNX model to `models/attack_classifier.onnx`.

---

### Mode 4: Physical CCTV Robustness Benchmark

Evaluate defense accuracy under simulated physical variations (viewing distances of 2m, 5m, 8m; viewing angles of 0°, 30°, 60°; and ambient, low-light, and backlit illumination):

```bash
python src/evaluation/benchmark.py --output reports/benchmark_report.md
```

This generates a comprehensive Markdown report with accuracy comparisons, precision, recall, and false positive rates.

---

## Running Automated Tests

Run the complete test suite with `pytest`:

```bash
# Run all 67 unit and integration tests
pytest tests/ -v

# Run video streaming and upload tests specifically
pytest tests/test_video_endpoints.py -v

# Run server REST and WebSocket tests
pytest tests/test_server.py -v

# Run classifier and latency tests
pytest tests/test_classifier.py -v
```

---

## API Reference (Quick Cheatsheet)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the DefOps monitoring dashboard |
| `GET` | `/api/v1/video-feed` | Continuous Motion-JPEG stream (`multipart/x-mixed-replace; boundary=frame`) |
| `POST` | `/api/v1/upload-video` | Upload `.mp4` or `.avi` video file to switch active stream source |
| `POST` | `/api/v1/stream-source` | Switch source dynamically (`{"source": 0, "loop": true}`) |
| `POST` | `/api/v1/stream-control` | Play/pause controls (`{"action": "play" \| "pause" \| "toggle"}`) |
| `GET` | `/api/v1/stream-status` | Current stream state, active source label, and counters |
| `POST` | `/api/v1/alerts` | Ingest an attack alert event from an inference worker |
| `GET` | `/api/v1/alerts` | Paginated query for historical alerts (`skip`, `limit`, `min_probability`) |
| `GET` | `/api/v1/metrics` | Retrieve current FPS, latency, detected persons, and flagged attackers |
| `POST` | `/api/v1/metrics` | Ingest runtime performance metrics updates |
| `WS` | `/ws/alerts` | Real-time WebSocket broadcasting alerts and live telemetry |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
