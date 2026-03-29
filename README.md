# 🎯 Face Detection: Multi-Mode Vision System

An intelligent, utility-based computer vision agent that calculates and tracks human faces across dynamic environments (live webcam feeds) and static state spaces (images and videos).

This project bridges classical machine learning (Haar feature-based cascade classifiers) with modern Python scripting to demonstrate real-time object detection, dynamic frame processing, and parameter optimization.

---

## 🧠 Core Computer Vision Concepts

This system is architected to explicitly demonstrate the foundational theories of computer vision object detection:

* **Feature Extraction & Classification:** The system does not rely on deep learning; instead, it utilizes a highly optimized **Haar Cascade Classifier**. It scans frames using edge, line, and four-rectangle features to identify human facial structures.
* **Dimensionality Reduction:** Before detection, the algorithm dynamically executes `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`. Converting the state space to grayscale drastically reduces computational weight, allowing for real-time Frames Per Second (FPS).
* **Algorithmic Optimization:** The agent utilizes mathematical tuning parameters (`scaleFactor=1.2` and `minNeighbors=6`) to rigorously balance detection accuracy against the computational cost of false positives.

---

## ⚙️ System Architecture: How It Works

The project adheres to strict Separation of Concerns (SoC) for high code quality:

1.  **The Interface (`main.py`):** The Python Command-Line Interface. It parses user input, manages the application loop, and handles I/O operations for reading local files and writing UI overlays.
2.  **The Inference Engine (`model/`):** Houses the pre-trained `haarcascade_frontalface_default.xml`. It acts as the offline knowledge base for facial geometries.
3.  **Dynamic Bounding & Annotation:** When a face is detected, Python dynamically injects spatial coordinates (`x, y, w, h`) and renders `cv2.rectangle` bounding boxes and FPS telemetry directly onto the output frame.

---

## 💻 Setup & Installation

**Prerequisites:**

* Python 3.8+
* OpenCV (`opencv-python`)

⚠️ **CRITICAL (Webcam Users):** Ensure your terminal or IDE has the necessary operating system permissions to access your integrated camera, otherwise the live stream will fail to initialize.

**1. Clone the Environment**

```bash
git clone https://github.com/tiwarirajat-1501/your-repo-name.git
cd your-repo-name
```

**2. Install the Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Execution & Tactical Scenarios

The system is strictly executed via the command line, launching an interactive terminal menu.

### Scenario 1: Real-Time Live Tracking (Dynamic Execution)

Forces the AI to open a local camera feed, continuously calculating bounding boxes and system FPS frame-by-frame.

```bash
python main.py
# Enter choice: 1
```

**Expected Terminal Output:**

```text
===== FACE DETECTION SYSTEM =====
1. Webcam Detection
2. Image Detection
3. Video Detection
4. Exit
Enter choice: 1
Press 'q' to quit
```

### Scenario 2: Static State Space (Image Analysis)

Calculates the absolute number of faces present in a single, unmoving environment.

```bash
python main.py
# Enter choice: 2
# Enter image path: screenshots/group_photo.jpg
```

**Expected Terminal Output:**

```text
===== FACE DETECTION SYSTEM =====
1. Webcam Detection
...
Enter choice: 2
Enter image path: screenshots/group_photo.jpg
✅ Faces detected: 4
```

---

## 📁 Repository Map

* `main.py` — Main execution script, menu logic, and CV bindings.
* `model/` — Directory containing the Haar Cascade XML inference engine.
* `requirements.txt` — Python dependencies (`opencv-python`).
* `screenshots/` — Directory containing visual nodal graphs and system UI demonstrations.
* `README.md` — Detailed system architecture, algorithmic analysis, and project documentation.

---

## 👨‍💻 Author

**Made by:** Rajat Tiwari
