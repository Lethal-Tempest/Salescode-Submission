# Screen Recapture Liveness Detector

* **Author**: Daksh Arora
* **Core Model**: ResNet101 (CNN Classifier)
* **Status**: Production Ready (91.0% Validation Accuracy)

This repository contains a deep learning-based liveness detector to determine whether a given image is a **genuine, real-world photo** or a **recaptured photo of a screen** (to detect spoofing/fraud in mobile app uploads).

---

## 1. Requirements

The project has been tested with Python 3.10+ and the following packages:
* `torch` (PyTorch)
* `torchvision`
* `Pillow`
* `Flask`

Install requirements:
```powershell
pip install torch torchvision pillow flask
```

---

## 2. Running the Predictor CLI

Wrap it in a one-line predictor as requested in the assignment:
```powershell
python predict.py <path_to_image>
```
* **Output**: A single float score between `0.00` and `1.00` representing the probability of the image being a screen recapture (e.g. `0.99` for screen, `0.06` for real).

---

## 3. Running the Live Webcam Demo

To start the interactive, real-time webcam liveness check:
```powershell
python app.py
```
Then open your browser at **`http://localhost:5000`**. 

* The application includes a beautiful glassmorphic UI, real-time scanning feedback, and a hardware device execution indicator.
* It includes a compatibility check that automatically falls back to CPU execution if GPU driver/architecture compatibility issues (like NVIDIA Blackwell `sm_120`) are encountered.

---

## 4. Assignment Deliverables
All required submission deliverables outlined in `ASSIGNMENT.pdf` are consolidated in:
* **[SUBMISSION_NOTE.md](SUBMISSION_NOTE.md)**: Contains average latency scores, on-device vs. cloud hosting cost calculations, and answers to core engineering questions (handling cheaters, mobile optimizations, and threshold choices).
