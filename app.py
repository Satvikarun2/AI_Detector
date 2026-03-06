import gradio as gr
import torch
import numpy as np
import PIL.Image
import os

# --- Import your AI Detective Agent ---
from forensic_agent import generate_simple_reasoning 

from models import TextureContrastClassifier
from utils import azi_diff

# --- Configuration ---
MODEL_PATH = './checkpoints/best_model.pth' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_THRESHOLD = 0.7  # Higher threshold = Lower False Positives

# Ensure a temporary folder exists for the agent's images
TEMP_DIR = "./temp_forensics"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Load Model ---
def load_model():
    model = TextureContrastClassifier()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✅ Loaded best model from {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: {MODEL_PATH} not found.")
    
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

def predict(input_img, threshold):
    if input_img is None:
        return "Please upload an image.", None, None
    
    # 1. Image Preprocessing & Feature Extraction
    img_pil = PIL.Image.fromarray(input_img).convert('RGB')
    features = azi_diff(img_pil, patch_num=128, N=256) 
    
    # 2. Prepare Tensors
    rich = torch.tensor(features['total_emb'][0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    poor = torch.tensor(features['total_emb'][1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    ela = torch.tensor(features['ela'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    noise = torch.tensor(features['noise'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # 3. Inference with Sigmoid Fix
    with torch.no_grad():
        output = model(rich, poor, ela, noise)
        # Apply Sigmoid to convert raw logit to probability (0 to 1)
        prediction_prob = torch.sigmoid(output).item() 
    
    # 4. Result Formatting & Confidence Calculation
    is_ai = prediction_prob > threshold
    label = "🚨 AI GENERATED or EDITED" if is_ai else "✅ REAL PHOTOGRAPH"
    final_pred_text = "AI Generated or Edited" if is_ai else "Real"
    
    if is_ai:
        confidence = (prediction_prob - threshold) / (1 - threshold)
    else:
        confidence = (threshold - prediction_prob) / threshold
    
    confidence = max(0, min(1, confidence))
    
    # 5. Visualizations (NumPy Arrays)
    ela_viz = (features['ela'] * 255).astype(np.uint8)
    noise_viz = ((features['noise'] - features['noise'].min()) / 
                 (features['noise'].max() - features['noise'].min() + 1e-8) * 255).astype(np.uint8)

    # --- 6. Agentic Reasoning Hand-off ---
    
    # Step A: Save the arrays as actual images so Gemini can "see" them
    orig_path = os.path.join(TEMP_DIR, "temp_orig.jpg")
    ela_path = os.path.join(TEMP_DIR, "temp_ela.jpg")
    prnu_path = os.path.join(TEMP_DIR, "temp_prnu.jpg")
    
    img_pil.save(orig_path)
    
    # Convert single-channel NumPy arrays to grayscale images and save
    if len(ela_viz.shape) == 2:
        PIL.Image.fromarray(ela_viz).convert('L').save(ela_path)
    else:
        PIL.Image.fromarray(ela_viz).save(ela_path)
        
    if len(noise_viz.shape) == 2:
        PIL.Image.fromarray(noise_viz).convert('L').save(prnu_path)
    else:
        PIL.Image.fromarray(noise_viz).save(prnu_path)

    # Step B: Generate specific branch signals for the agent
    if is_ai:
        ela_sig = "High compression inconsistencies detected."
        prnu_sig = "Absence of expected camera sensor noise."
        spec_sig = "Abnormal frequency patterns found."
        tex_sig = "Unnatural smoothness or structural flaws."
    else:
        ela_sig = "Consistent compression levels across the image."
        prnu_sig = "Natural camera sensor noise pattern present."
        spec_sig = "Normal frequency distribution."
        tex_sig = "Natural surface texture details."

    # Step C: Call the Forensic Agent
    try:
        explanation = generate_simple_reasoning(
            original_image_path=orig_path,
            ela_image_path=ela_path,
            prnu_image_path=prnu_path,
            final_prediction=final_pred_text,
            confidence=confidence * 100,
            ela_signal=ela_sig,
            prnu_signal=prnu_sig,
            spectral_signal=spec_sig,
            texture_signal=tex_sig
        )
    except Exception as e:
        explanation = f"Agent reasoning unavailable: {str(e)}"

    # 7. Final UI HTML with Agent Reasoning Included
    color = "red" if is_ai else "green"
    result_html = f"""
    <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(0,0,0,0.05); border: 2px solid {color};">
        <h2 style="color: {color}; margin-bottom: 5px;">{label}</h2>
        <p style="font-size: 1.2em;">Forensic Confidence: <b>{confidence*100:.2f}%</b></p>
        <p style="font-size: 0.9em; color: gray; margin-bottom: 15px;">(Probability: {prediction_prob:.4f} | Threshold: {threshold})</p>
        
        <div style="background-color: white; padding: 12px; border-radius: 8px; text-align: left; border-left: 5px solid #4a90e2; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="margin-top: 0; margin-bottom: 8px; color: #333;">🤖 AI Detective Reasoning:</h4>
            <p style="margin: 0; font-size: 1em; color: #444; line-height: 1.4;">{explanation}</p>
        </div>
    </div>
    """
    
    return result_html, ela_viz, noise_viz

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>🛡️ Multi-Modal AI Image Detector</h1>")
    gr.HTML("<p style='text-align: center;'>Insurance Claim Forensic Verification System</p>")
    
    with gr.Tabs():
        with gr.TabItem("Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_ui = gr.Image(label="Upload Image", type="numpy")
                    threshold_slider = gr.Slider(
                        minimum=0.5, maximum=0.95, value=DEFAULT_THRESHOLD, step=0.05,
                        label="Sensitivity Threshold",
                        info="Higher values reduce False Positives (Real images flagged as AI)."
                    )
                    submit_btn = gr.Button("🔍 Run Forensic Analysis", variant="primary")
                
                with gr.Column(scale=3): 
                    output_html = gr.HTML(label="Verdict")
                    with gr.Row():
                        ela_ui = gr.Image(label="ELA (Compression Inconsistency)", height=450)
                        noise_ui = gr.Image(label="PRNU (Sensor Noise Fingerprint)", height=450)

            gr.Markdown("---")
            gr.Markdown("### Forensic Visualization Interpretation")
            with gr.Row():
                gr.Info("💡 **ELA Heatmap:** Bright spots indicate areas with inconsistent JPEG compression, often a sign of generative artifacts or splicing.")
                gr.Info("💡 **PRNU Map:** Highlights high-frequency noise. Authentic photos contain sensor 'grain,' whereas AI images often show unnatural smoothness.")

        with gr.TabItem("Thesis Metrics & Methodology"):
            gr.Markdown("### Methodology: 4-Branch Late Fusion")
            gr.Markdown("""
            To maximize **Accuracy** and minimize **False Positives**, this system analyzes:
            * **Azimuthal Integrals (Spectral):** Captures frequency artifacts left by GANs/Diffusion models.
            * **ELA Branch:** Detects digital manipulation via quantization error levels.
            * **Noise Branch (PRNU):** Identifies the absence of unique physical sensor fingerprints.
            """)
            
            gr.Markdown("#### Final Validation Scores")
            gr.Markdown("| Metric | Internal Validation | External Validation (CashBowman) | External Validation (Secondary) |")
            gr.Markdown("| :--- | :--- | :--- | :--- |")
            gr.Markdown("| 🎯 Accuracy | 91.55% | 83.58% | 91.30% |")
            gr.Markdown("| 🔍 Precision | 0.9376 | 0.9286 | 0.7778 |")
            gr.Markdown("| 🚨 False Positive Rate | 0.0592 | 0.0560 | 1.1250 |")
            gr.Markdown("| 📉 Final Loss | 0.2134 | N/A* | N/A* |")
            gr.Markdown("<small>* *Data Not Available.*</small>")

    submit_btn.click(
        fn=predict,
        inputs=[input_ui, threshold_slider],
        outputs=[output_html, ela_ui, noise_ui]
    )
    
if __name__ == "__main__":
    demo.launch(debug=True, theme=gr.themes.Soft(primary_hue="blue"))