"""
pipeline_test.py

Full system verification test.

Tests:
1. Environment loading
2. Model loading
3. Feature extraction
4. Forensic signal extraction
5. Prompt builder
6. Azure OpenAI connection
7. Full forensic agent pipeline
"""

import os
import numpy as np
import PIL.Image
import torch

from dotenv import load_dotenv

from models import TextureContrastClassifier
from utils import azi_diff
from forensic_signals import ForensicSignalExtractor
from reasoning_prompt import build_prompt_pair
from forensic_agent import ForensicAgent


print("\n==============================")
print("🔬 FORENSIC PIPELINE TEST")
print("==============================\n")


# --------------------------------------------------
# 1️⃣ Load environment variables
# --------------------------------------------------

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if endpoint and api_key and deployment:
    print("✅ Azure environment loaded")
else:
    print("⚠️ Azure environment missing (LLM test will be skipped)")


# --------------------------------------------------
# 2️⃣ Test model loading
# --------------------------------------------------

MODEL_PATH = "C:/Users/EMPJE5Z/OneDrive - Allianz/Desktop/llm_reasoning/ai_detector_module/AI_Detector/checkpoints/best_model.pth"

try:

    model = TextureContrastClassifier()

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()

    print("✅ Model loaded successfully")

except Exception as e:

    print("❌ Model load failed:", e)
    exit()


# --------------------------------------------------
# 3️⃣ Create dummy image
# --------------------------------------------------

print("\nGenerating synthetic test image...")

dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

img = PIL.Image.fromarray(dummy)

print("✅ Test image created")


# --------------------------------------------------
# 4️⃣ Feature extraction
# --------------------------------------------------

try:

    features = azi_diff(img, patch_num=128, N=256)

    print("✅ Feature extraction working")

    print("   rich shape:", features["total_emb"][0].shape)
    print("   poor shape:", features["total_emb"][1].shape)
    print("   ela shape:", features["ela"].shape)
    print("   noise shape:", features["noise"].shape)

except Exception as e:

    print("❌ Feature extraction failed:", e)
    exit()


# --------------------------------------------------
# 5️⃣ Compute forensic signals
# --------------------------------------------------

try:

    extractor = ForensicSignalExtractor(threshold=0.7)

    signals = extractor.extract(
        raw_logit=0.5,
        rich_spectral=features["total_emb"][0],
        poor_spectral=features["total_emb"][1],
        ela_map=features["ela"],
        prnu_map=features["noise"]
    )

    print("\n✅ Forensic signals computed")

    print("   probability:", signals.probability)
    print("   verdict:", signals.verdict)
    print("   spectral anomaly:", signals.spectral.anomaly_score)
    print("   ela splicing:", signals.ela.splicing_indicator)
    print("   prnu strength:", signals.prnu.strength_score)

except Exception as e:

    print("❌ Signal extraction failed:", e)
    exit()


# --------------------------------------------------
# 6️⃣ Build reasoning prompt
# --------------------------------------------------

try:

    prompts = build_prompt_pair(signals)

    print("\n✅ Prompt builder working")

    print("   system prompt length:", len(prompts["system"]))
    print("   user prompt length:", len(prompts["user"]))

except Exception as e:

    print("❌ Prompt builder failed:", e)
    exit()


# --------------------------------------------------
# 7️⃣ Full forensic agent test
# --------------------------------------------------

try:

    agent = ForensicAgent(
        checkpoint_path=MODEL_PATH,
        device="cpu",
        enable_llm=False   # disable LLM for pipeline test
    )

    report = agent.analyze(img)

    print("\n✅ Forensic agent inference working")

    print("   probability:", report.signals.probability)
    print("   verdict:", report.signals.verdict)
    print("   risk:", report.signals.risk_level)
    print("   report length:", len(report.report_text))

except Exception as e:

    print("❌ Agent inference failed:", e)
    exit()


# --------------------------------------------------
# 8️⃣ Azure OpenAI connection test
# --------------------------------------------------

if endpoint and api_key and deployment:

    try:

        agent_llm = ForensicAgent(
            checkpoint_path=MODEL_PATH,
            device="cpu",
            enable_llm=True
        )

        report = agent_llm.analyze(img)

        print("\n✅ Azure OpenAI working")

        print("   LLM used:", report.llm_used)

    except Exception as e:

        print("⚠️ Azure test skipped or failed:", e)


print("\n==============================")
print("🎯 PIPELINE TEST COMPLETE")
print("==============================\n")