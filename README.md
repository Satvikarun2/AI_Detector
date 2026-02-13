Multi-Modal AI Image Detector (Forensic Edition)

This repository contains a state-of-the-art forensic analysis system designed to detect AI-generated or digitally manipulated images, specifically optimized for **Insurance Claim Verification** (e.g., car accident fraud detection).

Unlike standard detectors, this system utilizes a **4-Branch Late Fusion Architecture** to analyze images in both the spatial and frequency domains.

## 🚀 Key Features

* **Multi-Modal Detection**: Combines Fourier Spectral Analysis, Error Level Analysis (ELA), and PRNU Sensor Fingerprinting.
* **Gradio Web Interface**: A user-friendly dashboard for insurance adjusters to upload images and view real-time forensic maps.
* **Explainable AI (XAI)**: Generates visual heatmaps to justify the "AI vs. Real" verdict, making the model's decision-making transparent.
* **Tunable Sensitivity**: Includes a threshold slider to minimize **False Positives**, critical for high-stakes insurance use cases.

## 🔬 Methodology & Architecture

The model implements a **TextureContrastClassifier** that processes four distinct forensic signatures simultaneously:

1. **Rich Spectral Branch**: Analyzes high-texture patches using the **Azimuthal Integral** of the 2D Fourier Transform.
2. **Poor Spectral Branch**: Focuses on low-texture areas to find frequency artifacts hidden in smooth surfaces.
3. **ELA Branch**: Detects JPEG compression inconsistencies, highlighting areas that may have been digitally "spliced" or edited.
4. **Noise Branch (PRNU)**: Analyzes high-frequency noise to detect the absence of a physical camera sensor's unique fingerprint.

## 📊 Performance Metrics

The model was trained on a balanced dataset of **224,000 images** and validated against unseen "out-of-distribution" data to ensure real-world generalization.

| Metric | Internal Validation | External (Unseen) Test |
| --- | --- | --- |
| **Accuracy** | **78.68%** | **72.47%** |
| **False Positive Rate** | **0.29** | **0.27** |

> **Note**: The application utilizes the **Epoch 77** checkpoint (`best_model.pth`), which achieved the optimal balance of accuracy and low false-alarm rates.

## 🛠️ Installation & Usage

1. **Clone the Repository**:
```bash
git clone https://github.com/Satvikarun2/AI_Detector.git
cd AI_Detector

```


2. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


3. **Run the Application**:
```bash
python app.py

```



## 📂 Project Structure

* `models.py`: Defines the 4-branch Attention-based Fusion network.
* `utils.py`: Contains the Fourier Transform and feature extraction logic.
* `app.py`: The Gradio-based web interface for live inference.
* `test_unseen.py`: Script used for external validation on new, unseen datasets.
* `resume_train.py`: Script for long-duration training management.


