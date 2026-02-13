import gradio as gr
import torch
import numpy as np
import PIL.Image
import os
from models import TextureContrastClassifier
from utils import azi_diff

# --- Configuration ---
MODEL_PATH = './checkpoints/best_model.pth' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_THRESHOLD = 0.7  # Higher threshold = Lower False Positives

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
    features = azi_diff(img_pil, patch_num=128, N=256) #
    
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
    
    # 4. Result Formatting
    is_ai = prediction_prob > threshold
    label = "🚨 AI GENERATED or EDITED" if is_ai else "✅ REAL PHOTOGRAPH"
    
    # Calculate logical confidence relative to threshold
    if is_ai:
        confidence = (prediction_prob - threshold) / (1 - threshold)
    else:
        confidence = (threshold - prediction_prob) / threshold
    
    # Clamp confidence between 0 and 1 to prevent weird percentages
    confidence = max(0, min(1, confidence))
    
    color = "red" if is_ai else "green"
    result_html = f"""
    <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: rgba(0,0,0,0.05); border: 2px solid {color};">
        <h2 style="color: {color}; margin-bottom: 5px;">{label}</h2>
        <p style="font-size: 1.2em;">Forensic Confidence: <b>{confidence*100:.2f}%</b></p>
        <p style="font-size: 0.9em; color: gray;">(Probability: {prediction_prob:.4f} | Threshold: {threshold})</p>
    </div>
    """
    
    # 5. Visualizations
    ela_viz = (features['ela'] * 255).astype(np.uint8)
    noise_viz = ((features['noise'] - features['noise'].min()) / 
                 (features['noise'].max() - features['noise'].min() + 1e-8) * 255).astype(np.uint8)

    return result_html, ela_viz, noise_viz

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
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
                
                # Increased scale from 2 to 3 for larger visual output
                with gr.Column(scale=3): 
                    output_html = gr.HTML(label="Verdict")
                    with gr.Row():
                        # Added height to make forensic maps larger
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
            
            # Placeholder for your Unseen Accuracy results
            gr.Markdown("#### Final Validation Scores")
            gr.Markdown("| Metric | Internal (Val) | External (Unseen) |")
            gr.Markdown("| :--- | :--- | :--- |")
            gr.Markdown("| Accuracy | 78.68% | 72.47% |")
            gr.Markdown("| False Positive Rate | 0.29 | 0.27 |")

    submit_btn.click(
        fn=predict,
        inputs=[input_ui, threshold_slider],
        outputs=[output_html, ela_ui, noise_ui]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
