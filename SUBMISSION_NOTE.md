# Submission Note: Spot the Fake Photo Detector

* **Author**: Daksh Arora
* **Date**: July 2, 2026
* **Assignment Project**: Moire Pattern & Recapture Detection System

---

## 1. Executive Summary

This project implements a high-performance **Recapture Detection System** to differentiate between genuine, real-world photos and recaptured photos (photos of screens showing an image). 

We utilize a **ResNet101 Convolutional Neural Network (CNN)** pre-trained on recapture spoofing datasets. ResNet101's deep architecture automatically learns hierarchical representations of screen textures (Moire grid lines), chromatic aberrations, and reflection borders, achieving a validation accuracy of **91.0%** (validated using Stratified 5-Fold Cross Validation).

---

## 2. Performance Metrics & Deliverables

### Latency
* **CLI Command Invocation (Cold Start)**: **~4.4 seconds** (includes Python process boot, PyTorch/Torchvision library imports, and loading the 170MB weights file from disk).
* **In-Memory Server Execution (Warm)**: **~80 ms** per image (average preprocessing and forward pass duration on a standard laptop CPU).
* **Device**: Laptop CPU (Intel/AMD) with automated GPU fallback capability.

### Cost Analysis

#### On-Device Execution (Recommended)
* **Cost**: **$0.00** (Free).
* **Details**: Exporting the ResNet101 model to mobile-native formats allows inference to run entirely on the user's smartphone NPU/CPU. This eliminates server costs, works offline, and keeps user data private.

#### Cloud Server Execution (Alternative)
If hosted on cloud infrastructure, the estimated execution costs are:

| Metric | AWS Lambda (1024MB RAM) | Dedicated VM (e.g., AWS EC2 t3.medium) |
| :--- | :--- | :--- |
| **Cost per 1,000 images** | ~$0.00187 | ~$0.00115 (At full utilization) |
| **Cost per 1,000,000 images**| ~$1.87 | ~$1.15 |
| **Assumptions** | 100ms execution time, $0.20/M requests, compute price $0.00001667/GB-s | $30/month server fee, fully-utilized at 10 requests/second |

---

## 3. Engineering Decisions & Future Improvements

### Keeping it accurate as cheaters adapt
1. **Dynamic Reflection Liveness**: Prompt the user to tilt or rotate their phone slightly while taking a photo. Real physical objects show shifting specular highlights and glare, whereas screen displays contain static glare textures baked into the frame.
2. **Active Display Prompts**: Render a randomized, unique color border or floating token on the mobile screen during capture. Checking if the correct color signature reflects on the target verifies a live capture binding.
3. **Continuous Fine-tuning**: Set up an active learning loop. Flag predictions in the uncertain region (e.g. 0.20 to 0.70 probability) for human auditing, and feed verified edge cases back into the training pipeline.

### Making it tiny and fast enough for a phone
1. **Post-Training Quantization**: Convert ResNet101 weights from FP32 to INT8. This reduces the model size from 170MB to **~42MB** and speeds up inference on mobile hardware with less than 1% accuracy degradation.
2. **Knowledge Distillation**: Distill ResNet101's learned representations into a mobile-friendly student model (e.g., MobileNetV3 or ShuffleNetV2), compressing the model size to **<15MB**.
3. **NPU/GPU Compilation**: Compile the model to **CoreML** (iOS) and **TensorFlow Lite** (Android) to run on-device, bringing latency down to **5–15 ms**.

### Choosing the cut-off score for flagging fraud
Based on the empirical distribution of probabilities predicted by the model, we define three fraud zones:
* **Auto-Pass (Probability < 0.20)**: High confidence real photo. Automatically proceed.
* **Manual Review (0.20 to 0.70)**: Borderline confidence. Flag for human auditing.
* **Auto-Block (Probability >= 0.70)**: High confidence screen recapture. Automatically block the submission.
