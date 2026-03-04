import torch
from models import TextureContrastClassifier

model = TextureContrastClassifier()
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location="cpu"))

print("Model loaded successfully")