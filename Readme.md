# 🔐 FaceLocking: Edge ArcFace Face Recognition with 5-Point Alignment

A lightweight, real-time face recognition system optimized for embedded and edge devices (Jetson Nano, Raspberry Pi, edge PCs). This project combines classical computer vision with modern deep learning to create an efficient, modular face recognition pipeline suitable for deployment in resource-constrained environments.

**Pipeline:** Camera → Haar Detection → MediaPipe Landmarks → 5-Point Alignment → ArcFace Embeddings → Recognition

The system prioritizes accuracy, speed, and modularity while maintaining a low computational footprint—perfect for edge AI applications. 

---

## Table of Contents

- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Why 5-Point Alignment?](#why-5-point-alignment)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
  - [Face Detection (Haar Cascade)](#face-detection-haar-cascade)
  - [Facial Landmark Detection (MediaPipe FaceMesh)](#facial-landmark-detection-mediapipe-facemesh)
  - [5-Point Face Alignment](#5-point-face-alignment)
  - [ArcFace Embedding (ONNX Runtime)](#arcface-embedding-onnx-runtime)
  - [L2 Normalization & Similarity](#l2-normalization--similarity)
- [Visualization & Demo Features](#visualization--demo-features)
- [Requirements](#requirements)
- [Installation & Run](#installation--run)
- [Usage / Controls](#usage--controls)
- [Target Use Cases](#target-use-cases)
- [Roadmap / Future Improvements](#roadmap--future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Quick Start

Get up and running in under 5 minutes:

```bash
# 1. Clone or download the repository
cd FaceLocking

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the face recognition pipeline
python -m src.embed
```

> **Note:** Ensure `models/embedder_arcface.onnx` is in the models/ directory.

---

## Project Overview

The pipeline performs these steps in real-time:

1. **Capture** frames from your camera
2. **Detect** faces using Haar Cascade classifier
3. **Extract** 5 facial landmarks with MediaPipe FaceMesh
4. **Align** face to canonical 112×112 pose using affine transformation
5. **Generate** identity embeddings using ArcFace ONNX model
6. **Normalize** embeddings to unit length (L2 normalization)
7. **Compute** cosine similarity and visualize results

**Design Goals:** Minimal computational overhead, educational clarity, and modular architecture suitable for resource-constrained edge devices.

## Key Features

✨ **Real-time Performance** - Face detection and embedding generation at 15-30 FPS on edge hardware  
🎯 **Accurate Alignment** - 5-point landmark-based affine alignment for stable, high-quality embeddings  
⚡ **Lightweight** - Optimized for CPUs, no GPU required (though accelerated with CUDA/TensorRT if available)  
📦 **Portable** - ONNX Runtime ensures compatibility across platforms (Windows, Linux, Jetson, Raspberry Pi)  
🔍 **Debuggable** - Real-time visualization of detection, landmarks, aligned faces, and embeddings  
🧩 **Modular** - Clean separation of concerns; easy to swap components or extend functionality

## Why 5-Point Alignment?

ArcFace and similar models expect consistently aligned, frontal faces. Using 5 landmarks (left eye, right eye, nose, left mouth corner, right mouth corner) allows us to:

- Reduce pose variation
- Normalize scale and rotation
- Improve embedding stability and recognition accuracy
- Keep computation minimal (important for embedded devices)

---

## System Architecture

Camera Frame  
↓  
Haar Face Detection  
↓  
MediaPipe FaceMesh (5-point extraction)  
↓  
5-Point Face Alignment (Affine Transform → 112×112)  
↓  
ArcFace ONNX Embedding (embedder_arcface.onnx)  
↓  
L2 Normalization  
↓  
Cosine Similarity + Visualization

---

## Project Structure

```
FaceLocking/
├── src/
│   ├── __init__.py
│   ├── embed.py              # Main pipeline: camera input → embeddings
│   ├── recognize.py          # Face recognition and matching
│   ├── enroll.py             # Enroll new users into database
│   ├── evaluate.py           # Evaluate system performance
│   ├── camera.py             # Camera capture abstraction
│   ├── detect.py             # Face detection utilities
│   ├── align.py              # Face alignment utilities
│   ├── landmarks.py          # Landmark detection and extraction
│   ├── lock.py               # Access control / locking logic
│   ├── haar_5pt.py           # Haar detection + 5-point alignment
│   └── __pycache__/          # Cached bytecode
│
├── models/                   # Pre-trained models directory
│   └── embedder_arcface.onnx # ArcFace embedder (download required)
│
├── requirements.txt          # Python dependencies
├── init_project.py           # Project initialization script
├── fix_push_no_large_file.ps1 # Git LFS helper script
└── README.md                 # This file
```

**Key Entry Points:**
- `python -m src.embed` — Run real-time face embedding pipeline
- `python -m src.enroll` — Enroll new faces into the system
- `python -m src.recognize` — Perform face recognition on stored embeddings

---

## Core Components

### Face Detection (Haar Cascade)
- Fast, classical detector (OpenCV)
- Provides a rough bounding box for where to run the landmark detector
- Low computational cost — ideal for real-time performance on edge devices

### Facial Landmark Detection (MediaPipe FaceMesh)
- High-precision landmark detector
- From the full mesh only 5 key points are extracted
- Provides stable landmark positions even with small head movements

### 5-Point Face Alignment
- Affine transformation that maps detected landmarks to a fixed template
- Produces 112×112 aligned RGB face images required by ArcFace
- Normalizes pose, scale, and rotation

### ArcFace Embedding (ONNX Runtime)
- Pretrained ArcFace model loaded via ONNX Runtime
- Input: aligned 112×112 RGB face
- Output: embedding vector (identity-preserving, discriminative)

### L2 Normalization & Similarity
- Embeddings are L2-normalized to unit length
- Cosine similarity computed as dot(embedding_1, embedding_2)
  - Values near 1.0 suggest the same identity
  - Lower values suggest different identities

---

## Visualization & Demo Features

The demo includes:
- Live face bounding box
- 5-point landmark overlay
- Real-time aligned face preview (112×112)
- Embedding heatmap visualization
- FPS counter
- Real-time similarity display between consecutive frames

These make the system both educational and debuggable.

---

## Requirements

- Python 3.10 or 3.11 (MediaPipe may be unstable on 3.12+)
- See `requirements.txt` for exact packages. Typical dependencies include:
  - opencv-python
  - mediapipe
  - onnxruntime
  - numpy
  - matplotlib (optional for heatmap visualization)

---

## Installation & Run

### Prerequisites
- **Python 3.10 or 3.11** (MediaPipe may have stability issues on 3.12+)
- A working camera or video file
- Pre-trained ArcFace ONNX model

### Setup Steps

#### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Typical dependencies include:**
- `opencv-python` — Image capture and processing
- `mediapipe` — Facial landmark detection
- `onnxruntime` — Running the ArcFace model
- `numpy` — Numerical computations
- `matplotlib` (optional) — Heatmap visualization

#### 3. Prepare ArcFace Model
Obtain the `embedder_arcface.onnx` model and place it in the `models/` directory. Popular sources:
- [InsightFace](https://github.com/deepinsight/insightface)
- [ONNX Model Zoo](https://github.com/onnx/models)

#### 4. Run the Demo
```bash
python -m src.embed
```

A live window will appear showing:
- ✓ Face detection bounding boxes
- ✓ 5-point landmarks overlay
- ✓ Aligned 112×112 face preview
- ✓ Real-time FPS counter
- ✓ Embedding similarity scores

---

## Usage / Controls

While the embedding pipeline is running:

| Key | Action |
|-----|--------|
| **Q** | Quit the application |
| **P** | Print embedding statistics and debug info to terminal |
| **S** | Save current frame and embedding |

**Troubleshooting:**
- If the camera doesn't open, check camera permissions and try a different camera index
- Adjust lighting for better face detection performance
- Multiple faces are detected but only the largest is processed in the current version

---

## Target Use Cases

- Embedded face recognition systems
- Edge AI identity verification
- Attendance and access control systems
- Research and learning about face recognition pipelines

---

## Roadmap / Future Improvements

- Replace Haar with a lightweight CNN detector (for better multi-face robustness)
- Add an embedding database + matching pipeline
- Quantize ArcFace model for reduced power & faster inference
- Support multi-face tracking
- Hardware acceleration backends (TensorRT, NNAPI, etc.)
- Packaging for specific platforms (Jetson, Raspberry Pi)

---

## License

This project is intended for educational and research purposes. See the `LICENSE` file for details. 

**ONNX models may be subject to their own licenses—always review and comply with the licensing terms of pre-trained models you use.**

---

## Acknowledgments

- ArcFace / InsightFace
- MediaPipe
- OpenCV
- ONNX Runtime

---