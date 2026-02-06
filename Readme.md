# 🔐 FaceLocking
### Edge-Optimized ArcFace Recognition & Activity Tracking

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-green.svg)](https://onnxruntime.ai/)
[![MediaPipe](https://img.shields.io/badge/Landmarks-MediaPipe-red.svg)](https://mediapipe.dev/)

`FaceLocking` is a premium, real-time face recognition and tracking system engineered for the edge. It combines **ArcFace** deep embeddings with **MediaPipe** landmark detection to provide not just identity verification, but a persistent "lock-on" experience with integrated activity monitoring.

---

## 📽️ The Experience

The system is designed to "lock" onto a specific identity. Once locked:
- **Persistent Tracking**: The system stays fixed on the target person even in a crowd.
- **Action Detection**: Detects blinks, smiles, and head movements in real-time.
- **Liveness Heuristics**: Passive monitoring of facial dynamics to ensure a "living" presence.
- **Activity Logging**: Automatically records time-stamped actions to local history files.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **🚀 Real-time Performance** | 15-30 FPS on standard CPUs (Jetson Nano/Raspberry Pi friendly). |
| **🛡️ Robust Lock-On** | Sophisticated state management that tolerates brief occlusions or recognition drops. |
| **🎯 5-Point Alignment** | Classical affine transformation to 112x112 for maximum ArcFace accuracy. |
| **📊 Action Monitoring** | Heuristic-based detection of blinks, smiles, and directional look. |
| **📦 Modular Design** | Clean separation between Detection, Alignment, Embedding, and State Logic. |

---

## 🛠️ System Architecture

Our optimized pipeline ensures minimal latency while maintaining high precision:

1.  **Capture**: High-speed camera frame acquisition via OpenCV.
2.  **Detection**: Haar Cascade rough localization (CPU efficient).
3.  **Refinement**: MediaPipe FaceMesh for high-fidelity 5-point landmark extraction.
4.  **Normalization**: Affine transformation to a canonical 112x112 pose.
5.  **Intelligence**: ArcFace (ONNX) generates a discriminative identity embedding.
6.  **Comparison**: Cosine similarity against a lightweight local identity database.
7.  **Lock Logic**: State-machine-based tracking and action recording.

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/KOlivier2119/FaceLocking.git
cd FaceLocking

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize & Run
```bash
# Enroll your face (take pictures)
python -m src.enroll

# Run the FaceLocking system
python -m src.lock
```

---

## 🎮 Interactive Controls

While the system is running, use these keys to interact:

| Key | Action |
| :---: | :--- |
| `Q` | **Quit** the application safely |
| `L` | **Release Lock** manually to search for a new target |
| `R` | **Reload DB** if you added new faces while running |
| `S` | **Save** current frame and analytical data |

---

## 📂 Project Structure

```text
FaceLocking/
├── src/
│   ├── lock.py           # 🧠 The Main Engine: State machine & Action Detection
│   ├── enroll.py         # 👤 User enrollment and database creation
│   ├── recognize.py      # 🔍 Core recognition & matching logic
│   ├── embed.py          # ⚡ ArcFace ONNX embedder implementation
│   ├── align.py          # 📐 Affine transformation & 5-point alignment
│   └── landmarks.py      # 📍 Landmark extraction via MediaPipe
├── data/
│   ├── db/               # Generated identity embeddings (.npz)
│   └── lock_history/     # Time-stamped activity logs (.txt)
├── models/               # ArcFace ONNX model location
└── requirements.txt      # Project dependencies
```

---

## 📈 Roadmap

- [ ] **CNN Detection**: Swap Haar for a lightweight CNN (e.g., UltraFace) for better robustness.
- [ ] **Hardware Acceleration**: native support for OpenVINO and TensorRT backends.
- [ ] **Web Interface**: A modern dashboard for remote monitoring and history viewing.
- [ ] **Multi-target Lock**: Ability to track and log multiple identities simultaneously.

---

## 📄 License & Acknowledgments

This project is licensed under the MIT License. Special thanks to the teams behind **ArcFace (InsightFace)**, **MediaPipe**, and **ONNX Runtime** for providing the core building blocks of modern edge AI.