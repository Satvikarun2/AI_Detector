import google.generativeai as genai
from PIL import Image
import os

# Secure import of your key
try:
    from api_secrets import GEMINI_KEY
    genai.configure(api_key=GEMINI_KEY)
except ImportError:
    raise ImportError("api_secrets.py not found. Please create it with your GEMINI_KEY.")

# Global variable to store the correct model so we don't look it up every time
AUTO_MODEL_NAME = None

def get_best_model():
    """Asks Google which models your API key has access to and picks the best one."""
    global AUTO_MODEL_NAME
    if AUTO_MODEL_NAME:
        return AUTO_MODEL_NAME

    try:
        available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Flash is on top to avoid the '429 Quota Exceeded' error on the Pro models
        priorities = [
            "models/gemini-2.5-flash",
            "models/gemini-1.5-flash",
            "models/gemini-2.5-pro",
            "models/gemini-1.5-pro-latest",
            "models/gemini-1.5-pro",
            "models/gemini-pro-vision", 
            "models/gemini-1.0-pro-vision-latest"
        ]

        for p in priorities:
            if p in available:
                AUTO_MODEL_NAME = p.replace("models/", "")
                print(f"✅ Auto-selected model: {AUTO_MODEL_NAME}")
                return AUTO_MODEL_NAME
                
        if available:
            AUTO_MODEL_NAME = available[0].replace("models/", "")
            return AUTO_MODEL_NAME
            
        return "gemini-1.5-flash"
    except Exception as e:
        print(f"Error checking models: {e}")
        return "gemini-pro-vision"

def generate_simple_reasoning(original_image_path, ela_image_path, prnu_image_path, 
                              final_prediction, confidence, ela_signal, prnu_signal, 
                              spectral_signal, texture_signal):
    
    model_name = "Unknown Model"
    
    try:
        # 1. Automatically get the right model
        model_name = get_best_model()
        model = genai.GenerativeModel(model_name)
        
        # 2. Load the images
        if not all(os.path.exists(p) for p in [original_image_path, ela_image_path, prnu_image_path]):
            return f"Unable to generate explanation: Required forensic images are missing.<br><br><span style='font-size: 0.9em; color: #222; font-weight: bold;'>Powered by: {model_name}</span>"
            
        img = Image.open(original_image_path)
        ela_img = Image.open(ela_image_path)
        prnu_img = Image.open(prnu_image_path)
        
        # 3. Construct the prompt with INLINE DARK CSS to override Gradio's dark mode
        prompt = f"""
        You are an expert digital image forensics system.
        I have provided three images: Original, ELA map (compression artifacts), and PRNU map (sensor noise).

        Our neural network classified this as '{final_prediction}' with {confidence:.1f}% confidence based on 4 forensic branches.
        Branch Readings:
        1. ELA (Compression Consistency): {ela_signal}
        2. PRNU (Sensor Fingerprint): {prnu_signal}
        3. Spectral (Frequency Analysis): {spectral_signal}
        4. Texture (Surface Details): {texture_signal}

        TASK:
        Provide a detailed, two-part explanation for the '{final_prediction}' verdict.

        FORMAT YOUR EXACT RESPONSE LIKE THIS USING HTML TAGS (DO NOT CHANGE THE CSS STYLES):
        <span style="color: #111111; font-weight: bold; font-size: 1.1em;">🕵️ Everyday Summary:</span><br>
        [Write exactly ONE conversational sentence explaining why the image is '{final_prediction}'. You MUST synthesize all four branch readings into this single sentence without using technical jargon. Reference visual cues from the maps.]<br><br>

        <span style="color: #111111; font-weight: bold; font-size: 1.1em;">⚙️ Technical Summary:</span><br>
        [Write a 2-sentence technical breakdown. The first sentence must explain the technical findings of the ELA and PRNU maps. The second sentence must explain the technical findings of the Spectral and Texture signals. Use appropriate forensic terminology to explain how these 4 branches combined justify the final verdict.]
        """

        # 4. Call the model
        response = model.generate_content([prompt, img, ela_img, prnu_img])
        explanation = response.text.strip()
        
        # --- Make the footer dark and bold so it is easy to read ---
        formatted_output = f"{explanation}<br><br><span style='font-size: 0.9em; color: #333333; font-weight: 600;'><i>🔍 Forensic analysis powered by: <span style='color: #0056b3;'>{model_name}</span></i></span>"
        return formatted_output
        
    except Exception as e:
        return f"Agent Error (Auto-selected '{model_name}'): {str(e)}<br><br><span style='font-size: 0.9em; color: #333333; font-weight: bold;'><i>Attempted to use: {model_name}</i></span>"

if __name__ == "__main__":
    print("Testing Auto-Detect Forensic Agent...")