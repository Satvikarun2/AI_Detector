Multi-Modal AI Image Detector (Forensic Edition)
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
This repository contains a state-of-the-art forensic analysis system designed to detect AI-generated or digitally manipulated images, specifically optimized for Insurance Claim Verification (e.g., car accident fraud detection).

Unlike standard detectors, this system utilizes a 4-Branch Late Fusion Architecture to analyze images across multiple signal domains, ensuring that generative artifacts invisible to the human eye are captured.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 Key Features
Multi-Modal Detection: Combines Fourier Spectral Analysis, Error Level Analysis (ELA), and PRNU Sensor Fingerprinting into a single diagnostic pipeline.

Gradio Web Interface: A user-friendly dashboard for insurance adjusters to upload images and view real-time forensic heatmaps.

Explainable AI (XAI): Generates visual maps (ELA and PRNU) to justify the "AI vs. Real" verdict, making the model's decision-making transparent.

High-Fidelity Tuning: Optimized to minimize False Positives, a critical requirement for high-stakes insurance audits and legal evidence.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔬 Methodology & Architecture
The model implements a TextureContrastClassifier that processes four distinct forensic signatures simultaneously through independent neural branches:

Rich Spectral Branch: Analyzes high-texture patches using the Azimuthal Integral of the 2D Fourier Transform.

Poor Spectral Branch: Focuses on low-texture areas to find frequency artifacts (like checkerboard patterns) hidden in smooth surfaces.

ELA Branch: Detects JPEG compression inconsistencies, highlighting areas that may have been digitally "spliced".

Noise Branch (PRNU): Analyzes high-frequency noise to detect the absence of a physical camera sensor's unique fingerprint.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📊 Performance Metrics
The model underwent extensive training for 100 epochs on a massive dataset of 227,818 unique images. The final deployment utilizes the Epoch 100 checkpoint, which achieved superior generalization across multiple synthetic sources including DALL-E, CIFAKE, and Midjourney.

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/b5bc7737-4f4c-4f82-8dc8-03139b878dcb" />

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🛠️ Installation & Usage
1. Clone the Repository
Bash
git clone https://github.com/Satvikarun2/AI_Detector.git
cd AI_Detector
2. Install Dependencies
Bash
pip install -r requirements.txt
3. Run the Application
Bash
python app.py

📂 Project Structure
models.py: Defines the 4-branch Attention-based Fusion network.

utils.py: Contains the Fourier Transform and PRNU feature extraction logic.

app.py: The Gradio-based web interface for live inference.

test_unseen.py: Script used for external validation on new, unseen datasets.

resume_train.py: Script for long-duration training management (up to 100 epochs).

